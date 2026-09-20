/**
 * soundEngine.js - Multi-Tier Web Audio Synthesizer for Safety Alarms
 * Generates distinct, synthesized emergency sound effects with zero external file dependencies:
 *   1. Fire Siren (Hỏa hoạn / Khói): Âm còi hú quét tần số 700Hz - 1100Hz
 *   2. Medical Alert (Cấp cứu ngã): 3 hồi bíp dồn dập 900Hz
 *   3. Warning Chime (Vi phạm PPE): Âm chuông 2 nhịp nhắc nhở 660Hz -> 440Hz
 */

class SoundEngine {
  constructor() {
    this.ctx = null
    this.enabled = true
    this.volume = 0.8
  }

  initContext() {
    if (!this.ctx && typeof window !== 'undefined') {
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      if (AudioCtx) {
        this.ctx = new AudioCtx()
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {})
    }
  }

  setEnabled(val) {
    this.enabled = !!val
    if (this.enabled) {
      this.initContext()
    }
  }

  setVolume(val) {
    this.volume = Math.max(0, Math.min(1, Number(val)))
  }

  playAlert(type) {
    if (!this.enabled) return
    this.initContext()
    if (!this.ctx) return

    const t = String(type || '').toLowerCase()
    if (t.includes('fire') || t.includes('smoke') || t.includes('cháy') || t.includes('khói')) {
      this.playFireSiren()
    } else if (t.includes('fall') || t.includes('ngã')) {
      this.playMedicalAlert()
    } else {
      this.playWarningChime()
    }
  }

  /**
   * 1. Còi hú hỏa hoạn (Fire Alarm Siren): Quét tần số 750Hz <-> 1150Hz
   */
  playFireSiren() {
    try {
      const now = this.ctx.currentTime
      const osc = this.ctx.createOscillator()
      const gain = this.ctx.createGain()

      osc.type = 'sawtooth'
      osc.frequency.setValueAtTime(750, now)

      // Chu kỳ còi hú 1: Lên 1150Hz rồi hạ về 750Hz
      osc.frequency.linearRampToValueAtTime(1150, now + 0.45)
      osc.frequency.linearRampToValueAtTime(750, now + 0.90)
      // Chu kỳ còi hú 2
      osc.frequency.linearRampToValueAtTime(1150, now + 1.35)
      osc.frequency.linearRampToValueAtTime(750, now + 1.80)

      gain.gain.setValueAtTime(0.01, now)
      gain.gain.linearRampToValueAtTime(this.volume * 0.45, now + 0.1)
      gain.gain.setValueAtTime(this.volume * 0.45, now + 1.6)
      gain.gain.linearRampToValueAtTime(0.001, now + 1.85)

      osc.connect(gain)
      gain.connect(this.ctx.destination)

      osc.start(now)
      osc.stop(now + 1.9)
    } catch (e) {
      console.warn('Fire siren audio error:', e)
    }
  }

  /**
   * 2. Còi cấp cứu ngã (Medical Emergency Alert): 3 hồi bíp dồn dập 880Hz
   */
  playMedicalAlert() {
    try {
      const now = this.ctx.currentTime
      const duration = 0.1
      const gap = 0.07

      for (let i = 0; i < 3; i++) {
        const start = now + i * (duration + gap)
        const osc = this.ctx.createOscillator()
        const gain = this.ctx.createGain()

        osc.type = 'square'
        osc.frequency.setValueAtTime(880, start)
        osc.frequency.linearRampToValueAtTime(940, start + duration)

        gain.gain.setValueAtTime(0.01, start)
        gain.gain.linearRampToValueAtTime(this.volume * 0.4, start + 0.02)
        gain.gain.setValueAtTime(this.volume * 0.4, start + duration - 0.02)
        gain.gain.linearRampToValueAtTime(0.001, start + duration)

        osc.connect(gain)
        gain.connect(this.ctx.destination)

        osc.start(start)
        osc.stop(start + duration + 0.01)
      }
    } catch (e) {
      console.warn('Medical alert audio error:', e)
    }
  }

  /**
   * 3. Chuông nhắc nhở bảo hộ (PPE Warning Chime): 2 nhịp 660Hz -> 440Hz êm tai
   */
  playWarningChime() {
    try {
      const now = this.ctx.currentTime

      // Tone 1: 660Hz
      const osc1 = this.ctx.createOscillator()
      const gain1 = this.ctx.createGain()
      osc1.type = 'sine'
      osc1.frequency.setValueAtTime(660, now)
      gain1.gain.setValueAtTime(this.volume * 0.35, now)
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.35)
      osc1.connect(gain1)
      gain1.connect(this.ctx.destination)
      osc1.start(now)
      osc1.stop(now + 0.35)

      // Tone 2: 440Hz
      const osc2 = this.ctx.createOscillator()
      const gain2 = this.ctx.createGain()
      osc2.type = 'sine'
      osc2.frequency.setValueAtTime(440, now + 0.18)
      gain2.gain.setValueAtTime(this.volume * 0.35, now + 0.18)
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.6)
      osc2.connect(gain2)
      gain2.connect(this.ctx.destination)
      osc2.start(now + 0.18)
      osc2.stop(now + 0.6)
    } catch (e) {
      console.warn('Warning chime audio error:', e)
    }
  }
}

const soundEngine = new SoundEngine()
export default soundEngine
