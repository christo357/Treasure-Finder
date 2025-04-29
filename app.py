import streamlit as st
import cv2
import math
import random
import numpy as np
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
# os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

RESIZE_DIM = 320

# Helper functions
def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_round_score(elapsed_secs: float) -> int:
    """
    Example scoring functions — pick one that fits you:

    1) **Linear decay** (max 100 points, lose 1 point per second):
       score = max(0, 100 - elapsed_secs)

    2) **Inverse** (quick finds get big reward; scales down smoothly):
       score = int(1000 / (1 + elapsed_secs))

    3) **Exponential decay** (fast finds are heavily rewarded):
       base, decay = 500, 0.1
       score = int(base * math.exp(-decay * elapsed_secs))

    You can tweak the constants (100, 1000, base, decay) to taste.
    """
    # here’s the inverse version by default:
    base, decay = 500, 0.1
    score = int(base * math.exp(-decay * elapsed_secs))

    return score

def main():
    st.set_page_config(layout="wide")
    st.title("3D Object Tracking Treasure Hunt")
    col1, col2, col3 = st.columns([3,1,1])
    video_container = col1.empty()
    info_container = col2.empty()
    timer_container = col2.empty()
    score_container = col3.empty()
    thumbnail_container = col3.empty()

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
        st.session_state.prev_state = "WAIT_OBJECT"  # Add this to track state changes
        st.session_state.prev_time = datetime.datetime.now()
        st.session_state.frame_idx = 0
        
        # ⏱ timer & score
        st.session_state.start_time     = None
        st.session_state.end_time       = None
        st.session_state.timer_running  = False
        st.session_state.score          = 0
        st.session_state.thumbnails = [] 

    detector = st.session_state.detector
    tracker = st.session_state.tracker
    cap = st.session_state.cap
 
    if col2.button("Select Treasure"):
        st.session_state.reselect = True
        # reset timer
        st.session_state.start_time    = datetime.datetime.now()
        st.session_state.timer_running = True
        
    # select_btn = col2.button("Select Treasure")
    # if select_btn:
    #     st.session_state.reselect = True

    
    # button to reset (re-select) the hunter
    if col3.button("Reset Hunter"):
        st.session_state.reset_hunter = True
        # reset timer
        # st.session_state.start_time    = datetime.datetime.now()
        # st.session_state.timer_running = True
        
    
    if col3.button("Restart Game"):
        # Only save a thumbnail if a treasure was found in this round
        if st.session_state.state == "FOUND" and hasattr(st.session_state, 'last_frame'):
            # Create a thumbnail with the image and elapsed time
            thumbnail = {
                "image": st.session_state.last_frame,
                "elapsed": st.session_state.last_elapsed
            }
            st.session_state.thumbnails.append(thumbnail)
            st.toast(f"💾 Hunt saved! Completed in {st.session_state.last_elapsed:.2f} seconds")
            
        # Show a placeholder to clear the current video frame
        video_container.empty()
        info_container.empty()
        timer_container.empty()
        
        # Reset video capture if needed
        if hasattr(st.session_state, 'cap') :
            st.session_state.cap.release()
            del st.session_state.cap
        # st.session_state.cap = cv2.VideoCapture(0)
        
        time.sleep(0.5)
        
        # Create fresh VideoCapture instance
        st.session_state.cap = cv2.VideoCapture(0)
        
        for _ in range(5):
            if st.session_state.cap.isOpened():
                st.session_state.cap.read()
        
        
        # Reset game state but preserve thumbnails and score
        st.session_state.treasure_id = None
        st.session_state.treasure_centroid = None
        st.session_state.hunter_id = None
        st.session_state.state = "WAIT_OBJECT"
        st.session_state.start_time = None
        st.session_state.end_time = None
        st.session_state.timer_running = False
        st.session_state.frame_idx = 0
        st.session_state.known_obj_ids = set()
        
        # Clear the flag for next run
        st.session_state.reselect = False
        st.session_state.reset_hunter = False
        if hasattr(st.session_state, 'last_frame'):
            del st.session_state.last_frame
        if hasattr(st.session_state, 'last_elapsed'):
            del st.session_state.last_elapsed
        st.rerun()  # Force streamlit to rerun to update the UI
    
        

    # Reset Button
    if col3.button("Reset"):
        
        st.session_state.clear()
        st.rerun()
        # st.session_state.thumbnails = []  # Clear all thumbnails
        # st.session_state.score = 0  # Reset score
        
    if col3.button("❌ Exit Game", key="exit_game_btn"):
        st.session_state.cap.release()
        st.stop()  # Gracefully stops the app
    

    
    # Display thumbnails
    if st.session_state.thumbnails:
        thumbnail_container.markdown("### 🏆 Hunt History")
        
        # Create a scrollable area for thumbnails if there are many
        with thumbnail_container.container():
            for idx, thumbnail in enumerate(st.session_state.thumbnails):
                # Create a card-like display for each thumbnail
                st.markdown(f"**Hunt #{idx+1}**")
                
                # Display a small version of the image
                st.image(
                    thumbnail["image"], 
                    caption=f"Time: {thumbnail['elapsed']:.2f}s", 
                    width=150
                )
                
                # Add a separator between thumbnails
                if idx < len(st.session_state.thumbnails) - 1:
                    st.markdown("---")
    
    while cap.isOpened():
        # if col3.button("Reset Hunter"):
        #     st.session_state.hunter_id = None
        #     st.session_state.state = "WAIT_HUNTER" if st.session_state.treasure_id else "WAIT_OBJECT"

        
        ret, frame = cap.read()
        if not ret:
            info_container.warning("Cannot read from camera")
            break
        h0, w0 = frame.shape[:2]
        
        # Ensure the video feed restarts properly
        if st.session_state.state == "WAIT_OBJECT":
            info_container.info("Waiting for treasure selection...")

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

        # Check for state transition to HUNTING and start timer automatically
        if st.session_state.state == "HUNTING" and st.session_state.prev_state != "HUNTING":
            # Only start timer if it's not already running
            if not st.session_state.timer_running:
                st.session_state.start_time = datetime.datetime.now()
                st.session_state.timer_running = True
                st.toast("🕹️ Hunt begins!")

        # Update previous state for next iteration
        st.session_state.prev_state = st.session_state.state
        
        # Distance
        dist = 0
        if st.session_state.state == "HUNTING" and st.session_state.treasure_centroid:
            box_h = next((t.to_ltrb() for t in tracks if t.track_id==st.session_state.hunter_id and t.is_confirmed()), None)
            if box_h is not None:
                box_h = (box_h[0], box_h[1] + box_h[3]/2, box_h[2], box_h[3])
                c_h = centroid(box_h)
                c_t = st.session_state.treasure_centroid
                chx, chy = map(int, c_h)
                cv2.circle(frame, (chx, chy), 9, (255,0,0), -1)
                dist = euclidean(c_t, c_h)
                if dist < 50:
                    st.session_state.state = "FOUND"
                    
        if st.session_state.state == "FOUND" and st.session_state.timer_running:
            st.session_state.timer_running = False
            st.session_state.end_time      = datetime.datetime.now()
            # compute elapsed time
            elapsed = (st.session_state.end_time - st.session_state.start_time).total_seconds()
                
            # turn time into points
            round_pts = compute_round_score(elapsed)
            st.session_state.score += round_pts
            
            # Save the last frame in session state
            st.session_state.last_frame = frame.copy()  # Store the frame
            st.session_state.last_elapsed = elapsed  # Store the elapsed time

            # optionally show last-round points
            st.toast(f"🏅 You earned {round_pts} points!")


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
            # cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            # cv2.putText(frame, lbl, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if st.session_state.treasure_centroid:
            cx, cy = map(int, st.session_state.treasure_centroid)
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

        # Draw the moving hunter centroid
        # if st.session_state.hunter_id is not None:
        #     box_h = next((t.to_ltrb() for t in tracks if t.track_id == st.session_state.hunter_id and t.is_confirmed()), None)
        #     # if box_h is not None:
        #     #     # Calculate hunter center
        #     #     box_h = (box_h[0], box_h[1], box_h[2], box_h[3])  # Adjusting box center like you do in distance
        #     #     chx, chy = map(int, centroid(box_h))
        #     #     cv2.circle(frame, (chx, chy), 6, (255,0,0), -1)
        #     if c_h:
        #         # Calculate hunter center
        #         chx, chy = map(int, c_h)
        #         cv2.circle(frame, (chx, chy), 6, (255,0,0), -1)
            
            
        # Calculate max_dist dynamically based on treasure position
        if st.session_state.treasure_centroid:
            cx, cy = st.session_state.treasure_centroid
            
            # Calculate distances to each corner of the frame
            # dist_to_top_left = math.sqrt(cx**2 + cy**2)
            # dist_to_top_right = math.sqrt((w0 - cx)**2 + cy**2)
            # dist_to_bottom_left = math.sqrt(cx**2 + (h0 - cy)**2)
            # dist_to_bottom_right = math.sqrt((w0 - cx)**2 + (h0 - cy)**2)
            dist_to_right = w0-cx
            dist_to_left =  cx
            
            # Set max_dist to the maximum possible distance
            # max_dist = max(dist_to_top_left, dist_to_top_right, dist_to_bottom_left, dist_to_bottom_right)
            max_dist = max(dist_to_right, dist_to_left)
        else:
            # Fallback if treasure position not available
            max_dist = math.sqrt(w0**2 + h0**2) 
            
        # max_dist = 400
        frame_t = min(dist / max_dist, 1.0)

        # Interpolate between green (close) and red (far)
        r = int(255 * frame_t)
        g = int(255 * (1 - frame_t))
        b = 0

        # Apply translucent overlay BEFORE displaying
        overlay = np.full_like(frame, (b, g, r), dtype=np.uint8)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)  # 🛠 assign to frame

        # 🔵 Now show the updated tinted frame    

        video_container.image(frame, channels="BGR")
        info_container.write(f"Distance: {dist:.1f}px")
        
        
        # ── display timer ──
        if st.session_state.start_time:
            if st.session_state.timer_running:
                delta = datetime.datetime.now() - st.session_state.start_time
            else:
                delta = st.session_state.end_time - st.session_state.start_time
            
            secs = delta.total_seconds()
            mins, secs_int = divmod(int(secs), 60)
            micros = delta.microseconds
            
             # interpolate RGB
            MAX_TIME = 60.0
            time_t     = min(secs / MAX_TIME, 1.0)
            r     = int(255 * time_t)
            g     = int(255 * (1 - time_t))
            color = f"#{r:02x}{g:02x}00"

            
            # display with the interpolated color
            timer_text = f"⏱ {mins:02d}m:{secs_int:02d}s:{micros:06d}µs"
            timer_container.markdown(
                f"<span style='font-size:20px; color:{color};'>{timer_text}</span>",
                unsafe_allow_html=True
            )

        # ── display score ──
        score_container.write(f"🏆 Score: {st.session_state.score}")

        if st.session_state.state == "FOUND":
            info_container.success("Treasure Found!")
            if not st.session_state.end_time:  # Only set end time once
                st.session_state.end_time = datetime.datetime.now()
            # break

    cap.release()

if __name__ == "__main__":
    main()
