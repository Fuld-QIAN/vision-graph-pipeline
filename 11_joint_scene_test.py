"""Audit cached perception, fill missing HybridNets frames, export previews/graphs.

Run on the server. Missing YOLO frame coverage is an error, not an empty scene.
All coordinates remain raw-frame pixels. No claim of lane identity is made.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EVENTS = {"33_J1": (1245, 1834), "33_P1": (1950, 2599)}
LABELS = ['road','sidewalk','building','wall','fence','pole','traffic light',
          'traffic sign','vegetation','terrain','sky','person','rider','car',
          'truck','bus','train','motorcycle','bicycle']
CLASSES = ['person','bicycle','car','motorcycle','bus','truck','train','rider']

def _hsv_mask(image, ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h1,h2,s1,s2,v1,v2 in ranges:
        mask |= cv2.inRange(hsv,(h1,s1,v1),(h2,s2,v2))
    return mask

def _color_components(mask, area_min, area_max):
    """Use the same connected-component filters as stage 04."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result=[]
    for x,y,w,h,area in stats[1:]:
        if area_min <= int(area) <= area_max:
            result.append({'x1':float(x),'y1':float(y),'x2':float(x+w),'y2':float(y+h),'area':float(area)})
    return result

def _bbox_gap(first, second):
    dx=max(first['x1']-second['x2'],second['x1']-first['x2'],0.0)
    dy=max(first['y1']-second['y2'],second['y1']-first['y2'],0.0)
    return float((dx*dx+dy*dy)**0.5)

def detect_fixation_box(image, fixation_config=None):
    """Port stage 04 red/blue pairing; return at most one fixation box."""
    cfg = fixation_config or {'enabled':True,'red_hsv_ranges':[[0,10,80,255,80,255],[170,180,80,255,80,255]],
        'blue_hsv_ranges':[[90,140,60,255,60,255]],'red_component_area_min':80,'red_component_area_max':1000,
        'blue_component_area_min':30,'blue_component_area_max':500,'max_pair_distance_px':6,'bbox_padding_px':3,
        'marker_bbox_min_width_px':8,'marker_bbox_max_width_px':40,'marker_bbox_min_height_px':8,
        'marker_bbox_max_height_px':45,'max_markers_per_frame':1}
    if not cfg.get('enabled',True): return {'status':'disabled','box':None}
    red=_color_components(_hsv_mask(image,cfg.get('red_hsv_ranges',[])),int(cfg.get('red_component_area_min',20)),int(cfg.get('red_component_area_max',1000)))
    blue=_color_components(_hsv_mask(image,cfg.get('blue_hsv_ranges',[])),int(cfg.get('blue_component_area_min',10)),int(cfg.get('blue_component_area_max',500)))
    candidates=[]
    for ri,r in enumerate(red):
        for bi,b in enumerate(blue):
            gap=_bbox_gap(r,b)
            if gap <= float(cfg.get('max_pair_distance_px',30)):
                x1=max(0,min(r['x1'],b['x1'])-float(cfg.get('bbox_padding_px',3)))
                y1=max(0,min(r['y1'],b['y1'])-float(cfg.get('bbox_padding_px',3)))
                x2=min(float(image.shape[1]),max(r['x2'],b['x2'])+float(cfg.get('bbox_padding_px',3)))
                y2=min(float(image.shape[0]),max(r['y2'],b['y2'])+float(cfg.get('bbox_padding_px',3)))
                if (float(cfg.get('marker_bbox_min_width_px',8)) <= x2-x1 <= float(cfg.get('marker_bbox_max_width_px',40)) and
                    float(cfg.get('marker_bbox_min_height_px',8)) <= y2-y1 <= float(cfg.get('marker_bbox_max_height_px',45))):
                    candidates.append({'box':[int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))],
                        'pair_gap_px':round(gap,3),'red_area_px':int(r['area']),'blue_area_px':int(b['area']),
                        'red_component_index':ri,'blue_component_index':bi})
    if not candidates: return {'status':'unpaired','box':None,'red_components':len(red),'blue_components':len(blue)}
    chosen=max(candidates,key=lambda m:(m['blue_area_px'],m['red_area_px'],-m['pair_gap_px']))
    chosen.update(status='paired',red_components=len(red),blue_components=len(blue),marker_type='red_blue_gaze_marker')
    return chosen

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def load_fixations(path):
    rows = read_csv(path)
    grouped = {}
    for row in rows:
        event = str(row.get('event_id','')).strip()
        frame = frame_id(row)
        box = [float(pick(row, key+'_px')) for key in ('x1','y1','x2','y2')]
        if not event or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f'Invalid fixation row: {row}')
        grouped.setdefault((event,frame), []).append({
            'fixation_id': row.get('fixation_id',''), 'box': box,
            'marker_type': row.get('marker_type','red_blue_gaze_marker'),
            'pair_gap_px': row.get('pair_gap_px',''),
            'source': str(path),
        })
    return grouped

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def pick(row, *names):
    for name in names:
        if name in row and str(row[name]).strip():
            return row[name]
    raise ValueError(f'Missing column {names}; actual columns: {list(row)}')

def frame_id(row):
    value = float(pick(row, 'frame', 'frame_index', 'frame_id'))
    if not value.is_integer():
        raise ValueError('Non-integer frame number')
    return int(value)

def read_mask(path, shape, semantic=False):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.shape != shape or a.ndim != 2:
        raise ValueError(f'Invalid mask shape {path}: {a.shape}, expected {shape}')
    allowed = set(range(19)) | {255} if semantic else {0, 1, 255}
    if not set(np.unique(a)).issubset(allowed):
        raise ValueError(f'Invalid mask labels: {path}')
    return a if semantic else (a > 0).astype(np.uint8)

def color(key):
    return tuple(60 + b % 196 for b in hashlib.sha256(str(key).encode()).digest()[:3])

def normalize_track(row, width, height):
    # The full tracking table also retains detections that were not assigned a
    # track. They are valid audit records but must not become graph nodes.
    raw_tid = row.get('track_id', '')
    tid = str(raw_tid).strip()
    if not tid or tid in {'-1', 'None', 'nan', 'not_assigned'}:
        return None
    box = [float(pick(row, key+'_px', key)) for key in ('x1','y1','x2','y2')]
    x1,y1,x2,y2 = box
    if not np.isfinite(box).all() or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'Invalid raw-pixel box: {box}')
    name = str(pick(row, 'class_name', 'label', 'category')).lower()
    confidence = float(pick(row, 'confidence', 'score', 'conf'))
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid confidence')
    overlap = str(row.get('fixation_overlap', row.get('overlaps_fixation', 'unknown')))
    return {'track_id':tid, 'class_name':name, 'box':box, 'confidence':confidence, 'fixation_overlap':overlap}

def box_iou(a,b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    union=max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    return inter/union

def load_pose_model(weights, device):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('未安装 ultralytics，无法加载行人姿态模型。') from error
    return YOLO(str(weights)), device

def attach_pose(image, tracks, pose_runtime, threshold=.10):
    for track in tracks: track['pose']=None
    if pose_runtime is None: return {'status':'disabled','detections':0,'matched':0}
    model,device=pose_runtime
    result=model.predict(source=image,device=device,verbose=False,conf=.15,iou=.5,max_det=50)[0]
    if result.keypoints is None or result.boxes is None: return {'status':'ok','detections':0,'matched':0}
    boxes=result.boxes.xyxy.cpu().numpy()
    points=result.keypoints.xy.cpu().numpy()
    conf=result.keypoints.conf
    conf=conf.cpu().numpy() if conf is not None else np.ones(points.shape[:2],np.float32)
    people=[]
    for box,xy,cf in zip(boxes,points,conf):
        valid=np.isfinite(xy).all(axis=1)&(cf>=.20)
        people.append({'box':box.astype(float).tolist(),'keypoints':xy.astype(float).tolist(),
                       'keypoint_confidence':cf.astype(float).tolist(),'valid_keypoints':int(valid.sum()),
                       'pose_confidence':float(np.mean(cf[valid])) if valid.any() else 0.0})
    pairs=[]
    for ti,t in enumerate(tracks):
        if t['class_name'] not in {'person','rider'}: continue
        for pi,p in enumerate(people): pairs.append((box_iou(t['box'],p['box']),ti,pi))
    used_t=set(); used_p=set(); matched=0
    for overlap,ti,pi in sorted(pairs,reverse=True):
        if overlap<threshold or ti in used_t or pi in used_p: continue
        tracks[ti]['pose']=people[pi]; tracks[ti]['pose']['match_iou']=round(float(overlap),6)
        used_t.add(ti); used_p.add(pi); matched+=1
    return {'status':'ok','detections':len(people),'matched':matched,'unmatched_pose':len(people)-len(used_p)}

def suppress_fixation_false_positive(track, fixation, width, height, max_area_ratio=.01):
    if fixation.get('box') is None or track['class_name'] not in {'person','rider'}:
        return False
    x1,y1,x2,y2=track['box']; fx1,fy1,fx2,fy2=fixation['box']
    center=((x1+x2)/2,(y1+y2)/2)
    area=(x2-x1)*(y2-y1)/(width*height)
    return fx1 <= center[0] <= fx2 and fy1 <= center[1] <= fy2 and area <= max_area_ratio

def graph(semantic, lane, tracks, frame, fps, event):
    h,w = semantic.shape
    ratios = [float(np.mean(semantic == i)) for i in range(19)]
    # Shared 37-column features: type(3), position/size/conf(5), class(9),
    # semantic fractions(19), lane fraction(1).
    scene = [1.,0.,0.] + [0.]*14 + ratios + [float(lane.mean())]
    nodes = [scene]
    ids = ['scene']
    edges = {(0,0)}
    count, _, stats, centers = cv2.connectedComponentsWithStats(lane, 8)
    # Connected components are fragments, not identified lanes.
    for i in range(1,count):
        x,y,bw,bh,area = stats[i]
        if area < 8:
            continue
        nodes.append([0.,1.,0.,float(centers[i,0]/w),float(centers[i,1]/h),
                      float(bw/w),float(bh/h),0.] + [0.]*9 + [0.]*19 + [float(area/(h*w))])
        ids.append(f'lane_fragment_{i}')
    for t in tracks:
        x1,y1,x2,y2 = t['box']
        onehot = [float(t['class_name'] == c) for c in CLASSES] + [float(t['class_name'] not in CLASSES)]
        nodes.append([0.,0.,1.,(x1+x2)/(2*w),y2/h,(x2-x1)/w,(y2-y1)/h,t['confidence']] + onehot + [0.]*20)
        ids.append('track_'+t['track_id'])
    for i in range(1,len(nodes)):
        edges.update({(i,i),(0,i),(i,0)})
        neighbors = sorted((j for j in range(1,len(nodes)) if j != i),
                           key=lambda j: (nodes[i][3]-nodes[j][3])**2 + (nodes[i][4]-nodes[j][4])**2)[:3]
        for j in neighbors:
            edges.update({(i,j),(j,i)})
    return {'event_id':event,'frame':frame,'time_s':frame/fps,'node_ids':ids,
            'x':nodes,'edges':sorted(edges),'tracks':tracks,
            'coordinate_system':'raw_frame_normalized','trained':False}

def run(args):
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {'status':'running','events':{},'errors':[], 'manual_validation':'pending',
               'time_status':'nominal_not_independently_verified','perspective_correction':False,
               'screen_mask':False,'graph_features':'type3,xywh_conf5,class9,semantic19,lane_ratio1',
               'frame_scope':'defined_event_window' if args.use_defined_event_ranges else 'frame_index'}
    config = json.loads(args.config.read_text(encoding='utf-8'))
    fixation_config = config.get('fixation_marker', {})
    pose_weights = getattr(args, 'pose_weights', None)
    pose_runtime = load_pose_model(pose_weights, args.device) if pose_weights else None
    dump(out/'summary.json', summary)
    module_path = Path(__file__).with_name('10_hybridnets_scene_fusion.py')
    spec = importlib.util.spec_from_file_location('fusion', module_path)
    fusion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion)
    runtime = None
    try:
        rows = read_csv(args.frame_index)
        requested_events = {
            item.strip() for item in str(getattr(args, 'event_id', '') or '').split(',')
            if item.strip()
        }
        available_events = sorted({str(row.get('event_id', '')).strip() for row in rows if str(row.get('event_id', '')).strip()})
        if requested_events:
            missing_events = requested_events - set(available_events)
            if missing_events:
                raise ValueError(f'frame_index 没有指定事件: {sorted(missing_events)}')
            event_order = [event for event in available_events if event in requested_events]
        else:
            event_order = available_events
        if not event_order:
            raise ValueError('frame_index 中没有可用 event_id')
        semantic_dir = args.semantic_dir
        for event in event_order:
            selected = [r for r in rows if r.get('event_id') == event]
            defined_range = None
            if args.use_defined_event_ranges:
                if event not in EVENTS:
                    raise ValueError(f'{event}: EVENTS does not define a frame range')
                first_frame, last_frame = EVENTS[event]
                defined_range = [first_frame, last_frame]
                selected = [r for r in selected if first_frame <= frame_id(r) <= last_frame]
            selected.sort(key=frame_id)
            selected_frames = [frame_id(r) for r in selected]
            if len(selected_frames) != len(set(selected_frames)):
                raise ValueError(f'{event}: frame_index 含重复帧号')
            if not selected_frames:
                raise ValueError(f'{event}: 没有可用帧')
            if defined_range:
                expected_frames = set(range(defined_range[0], defined_range[1] + 1))
                missing_frames = sorted(expected_frames - set(selected_frames))
                if missing_frames:
                    preview = missing_frames[:20]
                    raise ValueError(
                        f'{event}: defined event window is incomplete; '
                        f'missing {len(missing_frames)} frames, first missing={preview}'
                    )
            track_rows = read_csv(args.tracking_dir/f'tracked_detections_{event}.csv')
            stats_all = read_csv(args.tracking_dir/f'tracking_frame_stats_{event}.csv')
            selected_frame_set = set(selected_frames)
            for r in stats_all:
                if r.get('event_id',event) != event:
                    raise ValueError('Tracking statistics contain wrong event')
            stats = [r for r in stats_all if frame_id(r) in selected_frame_set]
            coverage = [frame_id(r) for r in stats]
            if len(coverage) != len(set(coverage)) or not set(selected_frames).issubset(coverage):
                raise ValueError(f'{event}: tracking coverage missing/duplicated; cannot infer zero detections')
            for r in stats:
                state = str(r.get('status','')).lower()
                if any(word in state for word in ('fail','error','missing','skip')):
                    raise ValueError(f'{event}: tracking frame reported {state}')
            grouped = {}
            for r in track_rows:
                if r.get('event_id',event) != event:
                    raise ValueError(f'{event}: wrong event in tracks')
                if frame_id(r) not in selected_frame_set:
                    continue
                grouped.setdefault(frame_id(r), []).append(r)
            # The tracker audit distinguishes raw/assigned/retained detections,
            # while the tracked CSV contains only rows that received a track
            # ID.  These counts are therefore not required to match.  Preserve
            # the discrepancy for review instead of treating it as corruption.
            tracking_count_discrepancies = []
            for r in stats:
                for key in ('assigned_detections','tracked_detections','assigned_count','tracked_count'):
                    if key in r and str(r[key]).strip():
                        declared = int(float(r[key]))
                        actual = len(grouped.get(frame_id(r),[]))
                        if declared != actual:
                            tracking_count_discrepancies.append({
                                'frame': frame_id(r), 'field': key,
                                'declared': declared, 'tracked_csv_rows': actual,
                                'interpretation': 'assigned_or_retained_count_differs_from_track_rows'
                            })
                        break
            fallback_fps = args.fps
            if not fallback_fps and selected[0].get('video_path'):
                video_path = Path(selected[0]['video_path'])
                if not video_path.is_absolute(): video_path = root/video_path
                cap = cv2.VideoCapture(str(video_path))
                fallback_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            fps_values = [float(r.get('fps') or r.get('video_fps') or fallback_fps or 0) for r in selected]
            if min(fps_values) <= 0 or not np.isfinite(fps_values).all() or max(fps_values)-min(fps_values)>1e-6:
                raise ValueError(f'{event}: provide verified container FPS using --fps; no FPS guessing')
            fps = fps_values[0]
            event_out = out/event
            event_out.mkdir()
            sheets = []
            sample_ids = set(np.linspace(0,len(selected)-1,20).round().astype(int))
            writer = None
            records = []
            graphs = []
            event_shape = None
            try:
                for pos,row in enumerate(selected):
                    f = frame_id(row)
                    record = {'event_id':event,'frame':f,'status':'failed'}
                    try:
                        image_path = Path(pick(row,'image_path'))
                        if not image_path.is_absolute(): image_path = root/image_path
                        image = fusion.load_image(image_path)
                        shape = image.shape[:2]
                        if event_shape is None: event_shape = shape
                        if event_shape != shape: raise ValueError('Frame dimensions changed within event')
                        sem_path = semantic_dir/'semantic_masks'/event/f'mask_{f:06d}.png'
                        sem = read_mask(sem_path,shape,True)
                        lane = None
                        lane_source = None
                        for cached in args.fusion_dir:
                            p = cached/'masks'/'lane_supported_by_either_road'/event/f'lane_supported_by_either_road_{f:06d}.png'
                            if p.exists():
                                try: lane = read_mask(p,shape); lane_source = str(p); break
                                except ValueError: pass
                        if lane is None:
                            if not args.fill_fusion: raise ValueError('Missing valid lane mask; use --fill-fusion')
                            if runtime is None:
                                opt = fusion.parser().parse_args(['--hybridnets-root',str(root/'third_party/HybridNets'),
                                    '--weights',str(root/'third_party/HybridNets/weights/hybridnets.pth'),
                                    '--output-dir',str(out/'fusion_fill'),'--device',args.device,'--precision','fp16' if args.device=='cuda' else 'fp32'])
                                runtime,_ = fusion.load_hybridnets(opt)
                            hr,hl,_ = fusion.infer_one(image,runtime,event,f,.25,.3)
                            lane = fusion.fuse_masks(sem,hr,hl)['lane_supported_by_either_road']
                            p = event_out/f'lane_{f:06d}.png'
                            fusion.save_image(p,lane*255)
                            lane_source = str(p)
                        tracks = [t for t in (normalize_track(r,shape[1],shape[0]) for r in grouped.get(f,[])) if t is not None]
                        fixation = detect_fixation_box(image, fixation_config)
                        suppressed_tracks=[]
                        effective_tracks=[]
                        suppression_ratio=float(fixation_config.get('suppression_max_bbox_area_ratio',.01))
                        for track in tracks:
                            if suppress_fixation_false_positive(track,fixation,shape[1],shape[0],suppression_ratio):
                                track['suppressed_reason']='fixation_overlap_small_person_or_rider'; suppressed_tracks.append(track)
                            else: effective_tracks.append(track)
                        pose_status = attach_pose(image, effective_tracks, pose_runtime)
                        if len({t['track_id'] for t in tracks}) != len(tracks): raise ValueError('Duplicate track_id in frame')
                        g = graph(sem,lane,effective_tracks,f,fps,event)
                        graphs.append(g)
                        tinted = image.copy()
                        for i in range(19): tinted[sem==i] = color(LABELS[i])
                        preview = cv2.addWeighted(image,.7,tinted,.3,0)
                        preview[lane>0] = (0,255,255)
                        if fixation['box'] is not None:
                            fx1,fy1,fx2,fy2 = fixation['box']
                            cv2.rectangle(preview,(fx1,fy1),(fx2,fy2),(255,0,255),3)
                            cv2.putText(preview,'fixation',(fx1,max(16,fy1-5)),0,.5,(255,0,255),1)
                        for t in suppressed_tracks:
                            x1,y1,x2,y2=map(round,t['box']); cv2.rectangle(preview,(x1,y1),(x2,y2),(0,0,255),2)
                            cv2.putText(preview,'suppressed person FP',(x1,max(y1-4,32)),0,.4,(0,0,255),1)
                        for t in effective_tracks:
                            x1,y1,x2,y2 = map(round,t['box'])
                            c = color(event+'_'+t['track_id'])
                            cv2.rectangle(preview,(x1,y1),(x2,y2),c,2)
                            label = t['class_name']+' #'+t['track_id']+' fix='+t['fixation_overlap']
                            cv2.putText(preview,label,(x1,max(y1-4,32)),0,.4,c,1)
                            if t.get('pose'):
                                for x,y in np.asarray(t['pose']['keypoints']):
                                    if np.isfinite(x) and np.isfinite(y): cv2.circle(preview,(int(x),int(y)),2,(0,255,0),-1)
                        record.update(status='ok',tracking_status='tracked' if tracks else 'no_assigned_tracks',
                                      track_count=len(effective_tracks),raw_track_count=len(tracks),
                                      suppressed_track_count=len(suppressed_tracks),lane_source=lane_source,semantic_source=str(sem_path),
                                      fixation=fixation,pose=pose_status)
                    except Exception as error:
                        record['error'] = str(error)
                        summary['errors'].append({'event_id':event,'frame':f,'error':str(error)})
                        if event_shape is None:
                            raise
                        preview = np.zeros((*event_shape,3),np.uint8)
                    cv2.putText(preview,f'{event} frame={f} {record["status"]}',(5,20),0,.5,(255,255,255),1)
                    if writer is None:
                        h,w = event_shape
                        writer = cv2.VideoWriter(str(event_out/'joint_preview.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))
                        if not writer.isOpened(): raise RuntimeError('VideoWriter unavailable')
                    writer.write(preview)
                    if pos in sample_ids: sheets.append(cv2.resize(preview,(480,300)))
                    records.append(record)
                    if (pos+1)%50==0: print(f'{event}: {pos+1}/{len(selected)}',flush=True)
            finally:
                if writer is not None: writer.release()
                dump(event_out/'frame_index.json',records)
                with (event_out/'fixation_boxes.csv').open('w',encoding='utf-8-sig',newline='') as handle:
                    writer_csv = csv.DictWriter(handle, fieldnames=['event_id','frame','fixation_status','x1_px','y1_px','x2_px','y2_px','pair_distance_px','red_components','blue_components'])
                    writer_csv.writeheader()
                    for record in records:
                        fixation = record.get('fixation', {})
                        box = fixation.get('box') or [None]*4
                        writer_csv.writerow({'event_id':event,'frame':record['frame'],
                            'fixation_status':fixation.get('status','frame_failed' if record['status'] != 'ok' else 'unpaired'),
                            'x1_px':box[0],'y1_px':box[1],'x2_px':box[2],'y2_px':box[3],
                            'pair_distance_px':fixation.get('pair_distance_px'),
                            'red_components':fixation.get('red_components'), 'blue_components':fixation.get('blue_components')})
                with (event_out/'graphs.jsonl').open('w',encoding='utf-8') as handle:
                    for g in graphs: handle.write(json.dumps(g,ensure_ascii=False,allow_nan=False)+'\n')
            # Some E0-E4 intervals can contain fewer than 20 frames.  Keep a
            # fixed 5x4 review sheet without assuming that 20 samples exist.
            if not sheets:
                raise RuntimeError(f'{event}: no preview samples were created')
            blank = np.zeros_like(sheets[0])
            padded_sheets = sheets + [blank.copy() for _ in range(20-len(sheets))]
            sheet = np.vstack([np.hstack(padded_sheets[i:i+4]) for i in range(0,20,4)])
            fusion.save_image(event_out/'contact_sheet.jpg',sheet)
            cap = cv2.VideoCapture(str(event_out/'joint_preview.avi'))
            decoded = 0
            while cap.read()[0]: decoded += 1
            cap.release()
            if decoded != len(selected): raise RuntimeError('Preview video frame-count validation failed')
            summary['events'][event] = {'requested':len(selected),'success':len(graphs),'fps':fps,
                                        'defined_frame_range':defined_range,
                                        'preview_frames':decoded,
                                        'tracking_count_discrepancies':len(tracking_count_discrepancies)}
            dump(event_out/'tracking_count_discrepancies.json',tracking_count_discrepancies)
        summary['status'] = 'completed_with_failures' if summary['errors'] else 'completed'
    except Exception as error:
        summary['status'] = 'blocked'
        summary['errors'].append({'error':str(error)})
        raise
    finally:
        dump(out/'summary.json',summary)
        dump(out/'legend.json',{'semantic_bgr':{k:color(k) for k in LABELS},'lane_bgr':[0,255,255]})
        with (out/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as h:
            csv.writer(h).writerow(['event_id','frame','track_id','issue','severity','reviewer'])
    if summary['errors']: raise SystemExit(2)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frame-index',type=Path,required=True)
    p.add_argument('--semantic-dir',type=Path,required=True)
    p.add_argument('--tracking-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--pose-weights',type=Path,default=None)
    p.add_argument('--fusion-dir',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--event-id', default='', help='可选：只处理逗号分隔的事件；默认处理 frame-index 中全部事件')
    p.add_argument('--use-defined-event-ranges', action='store_true',
                   help='只处理 EVENTS 中定义的闭区间，并要求区间内每一帧均存在')
    p.add_argument('--fill-fusion',action='store_true')
    p.add_argument('--fps',type=float,help='Verified container FPS, only used if absent from frame CSV')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    run(p.parse_args())
