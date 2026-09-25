//! Play a Westwood VQA movie: video in a window, soundtrack on the default
//! audio device. Space pauses, Left/Right seek five seconds, Esc or Q quits.
//!
//! Usage: player <vqa file> [scale]      (scale: 1, 2, or 4; default 2)

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};

use vqa::{FramePixelsRef, FrameRef, Frames, VQA};

/// The audio side of playback: keeps the stream alive and exposes the
/// position counter the video loop uses as its clock.
struct Audio {
    /// Dropping the stream stops playback.
    _stream: cpal::Stream,
    /// Playback position in device frames. The callback advances it while
    /// unpaused; seeking stores a new value.
    position: Arc<AtomicUsize>,
    /// Length of the resampled soundtrack in device frames.
    total_frames: usize,
    /// Device sample rate.
    rate: u32,
}

/// The master playback clock: the audio position when the movie has sound,
/// wall time otherwise.
enum Clock {
    Audio {
        position: Arc<AtomicUsize>,
        rate: u32,
    },
    Wall {
        start: Instant,
        paused_accum: Duration,
        paused_since: Option<Instant>,
    },
}

impl Clock {
    /// Seconds of playback elapsed (frozen while paused).
    fn now(&self) -> f64 {
        match self {
            Clock::Audio { position, rate } => {
                position.load(Ordering::Relaxed) as f64 / f64::from(*rate)
            }
            Clock::Wall {
                start,
                paused_accum,
                paused_since,
            } => {
                let end = paused_since.unwrap_or_else(Instant::now);
                end.duration_since(*start)
                    .saturating_sub(*paused_accum)
                    .as_secs_f64()
            }
        }
    }

    /// Jump the clock to `t` seconds. For the audio clock this also moves
    /// playback: the callback reads samples at the position stored here.
    fn set(&mut self, t: f64) {
        match self {
            Clock::Audio { position, rate } => {
                position.store((t * f64::from(*rate)) as usize, Ordering::Relaxed);
            }
            Clock::Wall {
                start,
                paused_accum,
                paused_since,
            } => {
                let end = paused_since.unwrap_or_else(Instant::now);
                *paused_accum = Duration::ZERO;
                *start = end.checked_sub(Duration::from_secs_f64(t)).unwrap_or(end);
            }
        }
    }

    /// Wall-clock pause bookkeeping. The audio clock freezes on its own: the
    /// callback stops advancing the counter while the shared flag is set.
    fn set_paused(&mut self, paused: bool) {
        if let Clock::Wall {
            paused_accum,
            paused_since,
            ..
        } = self
        {
            if paused {
                *paused_since = Some(Instant::now());
            } else if let Some(since) = paused_since.take() {
                *paused_accum += since.elapsed();
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        println!("usage: {} <vqa file> [scale]", args[0]);
        return;
    }
    let scale = match args.get(2).map(String::as_str) {
        None | Some("2") => Scale::X2,
        Some("1") => Scale::X1,
        Some("4") => Scale::X4,
        Some(other) => {
            println!("bad scale {other:?}: expected 1, 2, or 4");
            return;
        }
    };

    let buffer = std::fs::read(&args[1]).expect("failed to read file");
    let vqa = VQA::parse(&buffer).expect("failed to parse VQA");
    println!("{:#?}", vqa.header);

    let width = usize::from(vqa.header.width);
    let height = usize::from(vqa.header.height);
    let title = format!("player - {}", args[1]);

    // Open the window before starting audio, so slow window creation can't
    // eat into the soundtrack.
    let mut window = Window::new(
        &title,
        width,
        height,
        WindowOptions {
            scale,
            ..WindowOptions::default()
        },
    )
    .expect("failed to open window");
    window.set_target_fps(60);

    let paused = Arc::new(AtomicBool::new(false));
    let audio = start_audio(&vqa, paused.clone());
    let mut clock = match &audio {
        Some(a) => Clock::Audio {
            position: a.position.clone(),
            rate: a.rate,
        },
        None => Clock::Wall {
            start: Instant::now(),
            paused_accum: Duration::ZERO,
            paused_since: None,
        },
    };

    let fps = f64::from(vqa.header.frame_rate.max(1));
    let mut frames = vqa.frames().expect("bad video header");
    let mut checkpoints = Checkpoints::new(&frames, (CHECKPOINT_SECONDS * fps) as usize);
    let mut buf = vec![0u32; width * height];
    let mut next_frame = 0usize;
    let mut video_done = false;

    while window.is_open() {
        if window.is_key_down(Key::Escape) || window.is_key_down(Key::Q) {
            break;
        }
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            let now_paused = !paused.load(Ordering::Relaxed);
            paused.store(now_paused, Ordering::Relaxed);
            clock.set_paused(now_paused);
            if now_paused {
                window.set_title(&format!("{title} (paused)"));
            } else {
                window.set_title(&title);
            }
        }

        // Left/Right seek five seconds back/forward; works while paused too.
        let mut seek = 0.0;
        if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
            seek -= 5.0;
        }
        if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
            seek += 5.0;
        }
        if seek != 0.0 {
            let t = (clock.now() + seek).max(0.0);
            clock.set(t);
            // Frames build on the decoder state left by their predecessors,
            // so a backward seek resumes from the last checkpoint before the
            // target; the catch-up loop below does the rest. Seeking past the
            // end just ends playback.
            let target = (t * fps) as usize;
            if target < next_frame {
                (frames, next_frame) = checkpoints.restore(target);
                video_done = false;
            }
        }

        // Decode every frame that has come due; draw only the newest. If the
        // loop stalled, this also catches video back up to the clock (the
        // intermediate decodes are mandatory anyway - they carry codebook
        // state). Frames are borrowed from the decoder, never copied out.
        let target = (clock.now() * fps) as usize;
        while !video_done && next_frame <= target {
            checkpoints.save(next_frame, &frames);
            match frames.next_ref() {
                Some(frame) => {
                    let frame = frame.expect("failed to decode frame");
                    if next_frame == target {
                        fill_buffer(frame, &mut buf);
                    }
                    next_frame += 1;
                }
                None => video_done = true,
            }
        }

        // Re-upload every iteration, even between movie frames: minifb does
        // not repaint on expose events, and this is also what polls input.
        window
            .update_with_buffer(&buf, width, height)
            .expect("failed to update window");

        let audio_done = audio
            .as_ref()
            .is_none_or(|a| a.position.load(Ordering::Relaxed) >= a.total_frames);
        if video_done && audio_done {
            break;
        }
    }
}

/// Convert a decoded frame into minifb's 0RGB u32 pixel layout.
fn fill_buffer(frame: FrameRef<'_>, out: &mut [u32]) {
    match frame.pixels {
        FramePixelsRef::Indexed { pixels, palette } => {
            for (out, &i) in out.iter_mut().zip(pixels) {
                let [r, g, b] = palette.get(usize::from(i)).copied().unwrap_or([0, 0, 0]);
                *out = u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b);
            }
        }
        FramePixelsRef::HiColor { pixels } => {
            for (out, &p) in out.iter_mut().zip(pixels) {
                // scale each 5-bit channel to 8 bits
                let scale = |v: u32| v << 3 | v >> 2;
                let p = u32::from(p);
                *out = scale(p >> 10 & 31) << 16 | scale(p >> 5 & 31) << 8 | scale(p & 31);
            }
        }
    }
}

/// Seconds of video between decoder checkpoints.
const CHECKPOINT_SECONDS: f64 = 5.0;

/// Copies of the frame iterator, saved every few seconds of video as
/// playback reaches them. Frames build on the decoder state their
/// predecessors left, so a backward seek resumes from the nearest earlier
/// copy instead of decoding from the start. Each copy holds one decoder's
/// state: under a megabyte for a 640x400 HiColor movie.
struct Checkpoints<'a> {
    /// Frames between checkpoints.
    every: usize,
    /// `saved[i]` decodes frame `i * every` next.
    saved: Vec<Frames<'a>>,
}

impl<'a> Checkpoints<'a> {
    fn new(frames: &Frames<'a>, every: usize) -> Self {
        Checkpoints {
            every: every.max(1),
            saved: vec![frames.clone()],
        }
    }

    /// Save `frames`, which decodes `next_frame` next, if that's where the
    /// next checkpoint falls.
    fn save(&mut self, next_frame: usize, frames: &Frames<'a>) {
        if next_frame == self.saved.len() * self.every {
            self.saved.push(frames.clone());
        }
    }

    /// The last checkpoint at or before `frame`, and the frame it decodes
    /// next.
    fn restore(&self, frame: usize) -> (Frames<'a>, usize) {
        let i = (frame / self.every).min(self.saved.len() - 1);
        (self.saved[i].clone(), i * self.every)
    }
}

/// Decode the soundtrack and start a cpal stream playing it. Returns `None`
/// if the movie has no sound or any part of audio setup fails (with a
/// warning where one is due); the caller then paces video by wall clock.
fn start_audio(vqa: &VQA<'_>, paused: Arc<AtomicBool>) -> Option<Audio> {
    if !vqa.header.has_sound() {
        return None;
    }
    let samples = match vqa.decode_audio() {
        Ok(samples) => samples,
        Err(e) => {
            eprintln!("warning: playing without audio: {e}");
            return None;
        }
    };
    if samples.is_empty() {
        return None;
    }
    // the playback path below wants interleaved stereo
    let samples = match vqa.header.num_channels() {
        1 => samples.iter().flat_map(|&s| [s, s]).collect(),
        _ => samples,
    };

    let host = cpal::default_host();
    let Some(device) = host.default_output_device() else {
        eprintln!("warning: playing without audio: no output device available");
        return None;
    };

    // Open the stream with the device's own default configuration - on some
    // hosts (e.g. WASAPI in shared mode) any other rate/format is rejected -
    // and adapt our audio to it instead.
    let supported = match device.default_output_config() {
        Ok(supported) => supported,
        Err(e) => {
            eprintln!("warning: playing without audio: {e}");
            return None;
        }
    };
    let config = supported.config();
    let rate = config.sample_rate;

    let samples = resample_stereo(&samples, vqa.header.sample_rate(), rate);
    let total_frames = samples.len() / 2;
    let position = Arc::new(AtomicUsize::new(0));

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => {
            build_stream::<f32>(&device, config, samples, position.clone(), paused)
        }
        cpal::SampleFormat::I16 => {
            build_stream::<i16>(&device, config, samples, position.clone(), paused)
        }
        cpal::SampleFormat::U16 => {
            build_stream::<u16>(&device, config, samples, position.clone(), paused)
        }
        format => {
            eprintln!("warning: playing without audio: unsupported sample format {format}");
            return None;
        }
    };
    let stream = match stream {
        Ok(stream) => stream,
        Err(e) => {
            eprintln!("warning: playing without audio: {e}");
            return None;
        }
    };
    if let Err(e) = stream.play() {
        eprintln!("warning: playing without audio: {e}");
        return None;
    }

    Some(Audio {
        _stream: stream,
        position,
        total_frames,
        rate,
    })
}

/// Build the output stream. Each device frame the callback reads the stereo
/// sample pair at the shared position counter and advances it - the master
/// clock for the video loop, and what seeking stores into. While `paused` is
/// set it emits silence without advancing, so the clock freezes; past the end
/// of the samples it keeps counting through silence so a video tail longer
/// than the soundtrack still gets paced.
fn build_stream<T: cpal::SizedSample + cpal::FromSample<i16>>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    samples: Vec<i16>,
    position: Arc<AtomicUsize>,
    paused: Arc<AtomicBool>,
) -> Result<cpal::Stream, cpal::Error> {
    let channels = usize::from(config.channels);

    device.build_output_stream(
        config,
        move |buffer: &mut [T], _: &cpal::OutputCallbackInfo| {
            // load once per callback so a mid-buffer toggle can't tear
            let is_paused = paused.load(Ordering::Relaxed);
            for frame in buffer.chunks_mut(channels) {
                let (left, right) = if is_paused {
                    (0, 0)
                } else {
                    let pos = position.fetch_add(1, Ordering::Relaxed);
                    match samples.get(pos * 2..pos * 2 + 2) {
                        Some(&[left, right]) => (left, right),
                        _ => (0, 0),
                    }
                };
                if let [out] = frame {
                    // mono device: mix both channels down
                    *out = T::from_sample(((i32::from(left) + i32::from(right)) / 2) as i16);
                } else {
                    for (ch, out) in frame.iter_mut().enumerate() {
                        *out = T::from_sample(match ch {
                            0 => left,
                            1 => right,
                            _ => 0,
                        });
                    }
                }
            }
        },
        |err| eprintln!("an error occurred on stream: {err}"),
        None,
    )
}

/// Resample interleaved stereo audio to a new rate using linear interpolation
fn resample_stereo(input: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }

    let in_frames = input.len() / 2;
    let out_frames = (in_frames as u64 * u64::from(to_rate) / u64::from(from_rate)) as usize;
    let mut out = Vec::with_capacity(out_frames * 2);
    for n in 0..out_frames {
        let pos = n as f64 * f64::from(from_rate) / f64::from(to_rate);
        let i = pos as usize;
        let frac = pos - i as f64;
        let next = (i + 1).min(in_frames - 1);
        for ch in 0..2 {
            let a = f64::from(input[i * 2 + ch]);
            let b = f64::from(input[next * 2 + ch]);
            out.push((a + (b - a) * frac) as i16);
        }
    }

    out
}
