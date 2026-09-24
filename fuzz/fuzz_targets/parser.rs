#![no_main]

use libfuzzer_sys::fuzz_target;
use vqa::*;

fuzz_target!(|data: &[u8]| {
    // Every public parser must fail cleanly on arbitrary input.
    let _ = vqa_version(data);
    let _ = frame_info(data);
    let _ = snd2_chunk(data);
    let _ = vqfr_chunk(data);
    let _ = cbf_chunk(data);
    let _ = vqa_header(data);
    let _ = finf_chunk(data);
    let _ = raw_chunk(data);
    let _ = cbp_chunk(data);
    let _ = cpl_chunk(data);
    let _ = vpt_chunk(data);
    let _ = vptr_chunk(data);
    let _ = vqfl_chunk(data);
    let _ = sn2j_chunk(data);

    // The high-level API must also hold up: parse, decode a bounded number
    // of video frames and convert them to RGB, and decode the soundtrack.
    // Borrowed frames (next_ref), skipped ones (nth), and a cloned iterator
    // must all agree with the owned frames.
    if let Ok(vqa) = VQA::parse(data) {
        if let (Ok(owned), Ok(mut borrowed)) = (vqa.frames(), vqa.frames()) {
            let mut frames = Vec::new();
            for frame in owned.take(16) {
                let view = borrowed.next_ref().expect("borrowed frames ended early");
                assert_eq!(frame.as_ref().map(Frame::view), view.as_ref().copied());
                if let Ok(frame) = &frame {
                    let _ = frame.to_rgb888();
                }
                let failed = frame.is_err();
                frames.push(frame);
                if failed {
                    break;
                }
            }
            // `frames` holds the first 16 frames, or all of them if the
            // movie ended or failed sooner, so it knows every nth below 16
            for n in [0, 3, 15] {
                let mut skipping = vqa.frames().expect("frames() succeeded above");
                assert_eq!(skipping.nth(n).as_ref(), frames.get(n));
            }
            if frames.len() > 2 {
                let mut original = vqa.frames().expect("frames() succeeded above");
                original.nth(1);
                let mut clone = original.clone();
                assert_eq!(original.next(), clone.next());
            }
        }
        let _ = vqa.decode_audio();
    }

    // Walk the container the way a real consumer does: FORM header, VQA
    // header, then FINF (scanning past any LINF/CINF chunks before it), and
    // finally the frame data each decoded FINF offset points at.
    let Ok((rest, _)) = form_chunk(data) else {
        return;
    };
    let Ok((rest, _)) = vqa_header(rest) else {
        return;
    };

    let Some(finf_pos) = rest.windows(4).position(|w| w == b"FINF") else {
        return;
    };
    let Ok((_, finf)) = finf_chunk(&rest[finf_pos..]) else {
        return;
    };

    for frame in finf.frames {
        let Some(frame_data) = data.get(frame.offset as usize..) else {
            continue;
        };
        let _ = snd2_chunk(frame_data);
        let _ = vqfr_chunk(frame_data);
        let _ = cbf_chunk(frame_data);
    }
});
