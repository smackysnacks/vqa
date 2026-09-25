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
    let every: usize = args
        .get(3)
        .map(|n| n.parse().expect("bad step"))
        .unwrap_or(1);

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
        let frame = frame.expect("failed to decode frame");
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
