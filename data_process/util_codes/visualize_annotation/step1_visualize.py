import json
import os
import cv2
import torch
from PIL import Image
from diffusers.utils import export_to_video

def draw_bbox_and_track_id_diffusers(video_path, json_data, output_path, cut, crop):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    s_x, e_x, s_y, e_y = crop
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame, end_frame = cut

    frames_for_export = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
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
            
            # Convert BGR frame to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames_for_export.append(pil_image)

        if frame_count > end_frame:
            break

        frame_count += 1
    
    cap.release()

    if frames_for_export:
        # Use diffusers' utility to save the list of PIL Images as a video
        export_to_video(frames_for_export, output_path, fps=fps)
        print(f"Processing completed, and the results have been saved to: {output_path}")
    else:
        print(f"No frames were processed in the specified range.")


if __name__ == "__main__":
    input_jsons_dir = "../../demo_result/step1/final_output/dataset1"
    video_root = "../../demo_result/step0/videos/dataset1"
    output_dir = "./step1/visual_video"

    os.makedirs(output_dir, exist_ok=True)

    jsons_files = [f for f in os.listdir(input_jsons_dir) if f.endswith(".json")]

    for json_file in jsons_files:
        with open(os.path.join(input_jsons_dir, json_file), "r") as f:
            json_data = json.load(f)

        metadata = json_data["metadata"]
        bbox = json_data["bbox"]

        video_path = os.path.join(video_root, metadata["path"])
        output_path = os.path.join(output_dir, json_file.replace(".json", ".mp4"))

        draw_bbox_and_track_id_diffusers(
            video_path, bbox, output_path, metadata["cut"], metadata["crop"]
        )
