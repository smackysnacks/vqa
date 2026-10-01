//! Decode a VQA movie's video frames and write them out as PPM images.
//!
//! Usage: dump_frames <vqa file> [out-dir] [every-nth]

use std::fs::File;
use std::io::Write;

use vqa::VQA;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("usage: {} <vqa file> [out-dir] [every-nth]", args[0]);
        return;
    }
    let out_dir = args.get(2).map(String::as_str).unwrap_or(".");
    let every: usize = match args.get(3).map(|n| n.parse()) {
        None => 1,
        Some(Ok(n)) if n > 0 => n,
        Some(_) => {
            println!("bad every-nth {:?}: expected a positive number", args[3]);
            return;
        }
    };

    let buffer = std::fs::read(&args[1]).expect("failed to read file");
    let vqa = VQA::parse(&buffer).expect("failed to parse VQA");
    println!(
        "{}x{} @ {} fps, {} frames, version {:?}",
        vqa.header.width,
        vqa.header.height,
        vqa.header.frame_rate,
        vqa.header.num_frames,
        vqa.header.version
    );

    // borrow each frame from the decoder and convert it into one reused
    // buffer; frames in between are still decoded (later frames build on
    // them) but never converted
    let mut frames = vqa.frames().expect("bad video header");
    let mut rgb = Vec::new();
    let mut i = 0;
    while let Some(frame) = frames.next_ref() {
        // a damaged movie gives up the frames before the bad one, and fails
        let frame = match frame {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        if i % every == 0 {
            rgb.resize(frame.width * frame.height * 3, 0);
            frame.write_rgb888(&mut rgb);
            let path = format!("{}/frame_{:04}.ppm", out_dir, i);
            let mut file = File::create(&path).expect("failed to create output file");
            write!(file, "P6\n{} {}\n255\n", frame.width, frame.height).unwrap();
            file.write_all(&rgb).unwrap();
            println!("wrote {}", path);
        }
        i += 1;
    }
}
