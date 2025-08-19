import json
import os
import cv2
from PIL import Image
from diffusers.utils import export_to_video

def draw_bbox_and_track_id(video_path, json_data, output_path, cut, crop):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    s_x, e_x, s_y, e_y = crop
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_for_export = []

    start_frame, end_frame = cut

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            # Crop the frame.
            frame = frame[s_y:e_y, s_x:e_x]
            frame_data = json_data.get(str(frame_count))

            if frame_data:
                for face in frame_data["face"]:
                    track_id = face["track_id"]
                    box = face["box"]
                    x1, y1, x2, y2 = (
                        int(box["x1"]),
                        int(box["y1"]),
                        int(box["x2"]),
                        int(box["y2"]),
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                    )

            # Convert the BGR frame to RGB and then to a PIL Image.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames_for_export.append(pil_image)

        if frame_count > end_frame:
            break

        frame_count += 1

    cap.release()
    
    # Check if any frames were collected before exporting.
    if frames_for_export:
        # Export the collected list of PIL Images to a video file.
        export_to_video(frames_for_export, output_path, fps=fps)
        print(
            f"Processing completed, only frames {start_frame} ~ {end_frame} were processed, and the results have been saved to: {output_path}"
        )
    else:
        print("No frames were processed in the specified range.")


if __name__ == "__main__":
    input_jsons_dir = "../../demo_result/step2/final_output/dataset1"
    video_root = "../../demo_result/step0/videos/dataset1"
    output_dir = "./step2/visual_video"

    os.makedirs(output_dir, exist_ok=True)

    jsons_files = [f for f in os.listdir(input_jsons_dir) if f.endswith(".json")]

    for json_file in jsons_files:
        with open(os.path.join(input_jsons_dir, json_file), "r") as f:
            json_data = json.load(f)

        metadata = json_data["metadata"]
        bbox = json_data["bbox"]

        video_path = video_root + "/" + metadata["path"]
        output_path = os.path.join(output_dir, json_file.replace(".json", ".mp4"))
        
        # Check for the correct key in metadata
        face_cut_key = 'face_cut' if 'face_cut' in metadata else 'cut'

        draw_bbox_and_track_id(
            video_path, bbox, output_path, metadata[face_cut_key], metadata["crop"]
        )
