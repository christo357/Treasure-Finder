import streamlit as st
import cv2
import math
import random
import numpy as np
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

RESIZE_DIM = 320

# Helper functions
def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def main():
    st.set_page_config(layout="wide")
    st.title("3D Object Tracking Treasure Hunt")
    col1, col2, col3 = st.columns([3,1,1])
    video_container = col1.empty()
    info_container = col2.empty()

    # Initialize or retrieve session states
    if 'detector' not in st.session_state:
        st.session_state.detector = YOLO('yolo11n.pt')
        st.session_state.tracker = DeepSort(max_age=30, n_init=3)
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.treasure_id = None
        st.session_state.treasure_centroid = None
        st.session_state.hunter_id = None
        st.session_state.known_obj_ids = set()
        st.session_state.state = "WAIT_OBJECT"
        st.session_state.prev_time = datetime.datetime.now()
        st.session_state.frame_idx = 0

    detector = st.session_state.detector
    tracker = st.session_state.tracker
    cap = st.session_state.cap

    start = col2.button("Select Treasure")
    if start:
        st.session_state.reselect = True
        
    # select_btn = col2.button("Select Treasure")
    # if select_btn:
    #     st.session_state.reselect = True

    reset_btn = col3.button("Reset Hunter")
   # button to reset (re-select) the hunter
    if reset_btn:
        st.session_state.reset_hunter = True
    
    while cap.isOpened():
        # if col3.button("Reset Hunter"):
        #     st.session_state.hunter_id = None
        #     st.session_state.state = "WAIT_HUNTER" if st.session_state.treasure_id else "WAIT_OBJECT"

        
        ret, frame = cap.read()
        if not ret:
            info_container.warning("Cannot read from camera")
            break
        h0, w0 = frame.shape[:2]

        # Detection / Tracking
        if st.session_state.frame_idx < 5 or st.session_state.frame_idx % 1 == 0:
            small = cv2.resize(frame, (RESIZE_DIM, RESIZE_DIM))
            results = detector(small)[0]
            scale_x = w0 / RESIZE_DIM
            scale_y = h0 / RESIZE_DIM
            dets = []
            for *box, conf, cls in results.boxes.data.tolist():
                x1, y1, x2, y2 = box
                x1o = int(x1 * scale_x)
                y1o = int(y1 * scale_y)
                wo = int((x2 - x1) * scale_x)
                ho = int((y2 - y1) * scale_y)
                dets.append(([x1o, y1o, wo, ho], conf, int(cls)))
            tracks = tracker.update_tracks(dets, frame=frame)
        else:
            tracks = tracker.update_tracks([], frame=frame)
        st.session_state.frame_idx += 1

        # Current objects and persons
        curr_objs = [t for t in tracks if t.is_confirmed() and t.det_class != 0]
        curr_persons = [t for t in tracks if t.is_confirmed() and t.det_class == 0]
        for t in curr_objs:
            st.session_state.known_obj_ids.add(t.track_id)

        # Reselection
        if st.session_state.get('reselect', False):
            if curr_objs:
                treasure_track = random.choice(curr_objs)
                st.session_state.treasure_id = treasure_track.track_id
                st.session_state.treasure_centroid = centroid(treasure_track.to_ltrb())
                st.session_state.state = "HUNTING" if st.session_state.hunter_id else "WAIT_HUNTER"
            st.session_state.reselect = False
            
        # ── Hunter reselection logic ──
        if st.session_state.get("reset_hunter", False):
            # if there are any tracked people, pick one at random
            if curr_persons:
                hunter_track = random.choice(curr_persons)
                st.session_state.hunter_id = hunter_track.track_id
                # update the state machine
                st.session_state.state = (
                    "HUNTING" if st.session_state.treasure_id else "WAIT_OBJECT"
                )
            # clear the flag so it only happens once
            st.session_state.reset_hunter = False

        # Auto assign
        if st.session_state.treasure_id is None and curr_objs:
            treasure_track = random.choice(curr_objs)
            st.session_state.treasure_id = treasure_track.track_id
            st.session_state.treasure_centroid = centroid(treasure_track.to_ltrb())
            st.session_state.state = "WAIT_HUNTER"
        if st.session_state.hunter_id is None and curr_persons:
            st.session_state.hunter_id = curr_persons[0].track_id
            if st.session_state.treasure_id:
                st.session_state.state = "HUNTING"

        # Distance
        dist = 0
        if st.session_state.state == "HUNTING" and st.session_state.treasure_centroid:
            box_h = next((t.to_ltrb() for t in tracks if t.track_id==st.session_state.hunter_id and t.is_confirmed()), None)
            if box_h is not None:
                box_h = (box_h[0], box_h[1] + box_h[3]/2, box_h[2], box_h[3])
                c_h = centroid(box_h)
                c_t = st.session_state.treasure_centroid
                dist = euclidean(c_t, c_h)
                if dist < 50:
                    st.session_state.state = "FOUND"

        # Visualization
        for t in tracks:
            if not t.is_confirmed(): continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            tid = t.track_id
            if tid == st.session_state.treasure_id:
                col = (0,255,0); lbl = f"T:{tid}"
            elif tid == st.session_state.hunter_id:
                col = (255,0,0); lbl = f"H:{tid}"
            else:
                col = (200,200,200); lbl = f"{t.det_class}:{tid}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, lbl, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if st.session_state.treasure_centroid:
            cx, cy = map(int, st.session_state.treasure_centroid)
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

        video_container.image(frame, channels="BGR")
        info_container.write(f"Distance: {dist:.1f}px")

        if st.session_state.state == "FOUND":
            info_container.success("Treasure Found!")
            break

    cap.release()

if __name__ == "__main__":
    main()
