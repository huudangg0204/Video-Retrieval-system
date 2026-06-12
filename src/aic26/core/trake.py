"""
TRAKE: Temporal Retrieval and Alignment of Key Events.

Finds sequences of keyframes that match event descriptions
in correct temporal order using dynamic programming.

Migrated from: core/trake.py
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from aic26.config import settings
from aic26.utils.frame_utils import parse_frame_int
from aic26.utils.media_utils import load_media_info
from aic26.core.search import SearchEngine
from aic26.core.rerank import Reranker


class TrakeEngine:
    """
    Temporal Retrieval of event sequences from videos.
    
    Two strategies:
    - trake_closest: Minimize total frame distance between events
    - trake_highest: Maximize total similarity score
    
    Events ending with '*' are reranked using BLIP ITM.
    
    Usage:
        trake = TrakeEngine(searcher, reranker)
        solutions = trake.trake_highest([
            "A player shooting",
            "A player dunking*",
            "Crowd cheering",
        ])
    """
    
    def __init__(
        self,
        searcher: SearchEngine,
        reranker: Reranker,
    ) -> None:
        """
        Initialize TRAKE engine.
        
        Args:
            searcher: SearchEngine for candidate retrieval
            reranker: Reranker for BLIP-based event reranking
        """
        self.searcher = searcher
        self.reranker = reranker
        logger.debug("TrakeEngine initialized")
    
    # ── Public API ────────────────────────────────────────────────
    
    def trake_closest(
        self,
        events: List[str],
        top_k: int = settings.trake_default_top_k,
        candidates: int = settings.trake_default_candidates,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find event sequences with minimum total frame gaps.
        
        Uses greedy DP: for each frame, keeps the highest-similarity
        candidate while preserving event order.
        
        Args:
            events: Event descriptions in temporal order
            top_k: Search results per event
            candidates: Number of candidate solutions
            
        Returns:
            List of solution dicts, or None if no valid sequence found
        """
        if not events:
            logger.warning("No events provided")
            return None
        
        logger.info(f"TRAKE closest: {len(events)} events")
        
        all_event_candidates, special_idx = self._gather_candidates(
            events, top_k, mode="closest"
        )
        candidate_videos = self._select_candidate_videos(
            all_event_candidates, special_idx
        )
        
        list_solutions: List[Tuple] = []
        
        for vid in candidate_videos:
            vid_candidates, all_present = self._collect_vid_candidates(
                vid, all_event_candidates
            )
            if not all_present:
                continue
            
            # Sort by frame, then by lag (event order)
            vid_candidates.sort(key=lambda x: (x["frame"], -x["lag"]))
            
            # Deduplicate: keep highest similarity per frame
            deduped: Dict[int, Dict] = {}
            for c in vid_candidates:
                f = c["frame"]
                if f not in deduped or deduped[f]["sim"] < c["sim"]:
                    deduped[f] = c
            vid_candidates = sorted(deduped.values(), key=lambda x: x["frame"])
            
            # Greedy DP
            D: Dict[int, Dict] = {}
            trace: Dict[int, Dict[int, Dict]] = defaultdict(dict)
            
            for cand in vid_candidates:
                lag = cand["lag"]
                if lag == 1:
                    D[lag] = cand
                else:
                    prev = D.get(lag - 1)
                    if prev is not None:
                        trace[lag - 1][cand["frame"]] = prev
                        D[lag] = cand
            
            list_solutions.extend(
                self._reconstruct_sequences(vid_candidates, events, trace, top_n=5)
            )
        
        return self._finalise_solutions(list_solutions, events, candidates)
    
    def trake_highest(
        self,
        events: List[str],
        top_k: int = settings.trake_default_top_k,
        candidates: int = settings.trake_default_candidates,
        option: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find event sequences with maximum total similarity.
        
        Uses optimized DP with score accumulation.
        Groups candidates by frame for efficient processing.
        
        Args:
            events: Event descriptions in temporal order
            top_k: Search results per event
            candidates: Number of candidate solutions
            option: 0 for standard search, 1 for threshold-based
            
        Returns:
            List of solution dicts, or None if no valid sequence found
        """
        if not events:
            logger.warning("No events provided")
            return None
        
        logger.info(f"TRAKE highest: {len(events)} events (option={option})")
        
        all_event_candidates, special_idx = self._gather_candidates(
            events, top_k, mode="highest", option=option
        )
        candidate_videos = self._select_candidate_videos(
            all_event_candidates, special_idx
        )
        
        list_solutions: List[Tuple] = []
        
        for vid in candidate_videos:
            vid_candidates, all_present = self._collect_vid_candidates(
                vid, all_event_candidates
            )
            if not all_present:
                continue
            
            vid_candidates.sort(key=lambda x: x["frame"])
            
            # Optimized DP
            dp_state: Dict[int, Dict] = {}
            dp_score: Dict[int, float] = defaultdict(float)
            trace: Dict[int, Dict[int, Dict]] = defaultdict(dict)
            
            i = 0
            while i < len(vid_candidates):
                cand = vid_candidates[i]
                frame = cand["frame"]
                
                # Collect all candidates at the same frame
                group = [cand]
                j = i + 1
                while j < len(vid_candidates) and vid_candidates[j]["frame"] == frame:
                    group.append(vid_candidates[j])
                    j += 1
                
                prev_state = dict(dp_state)
                prev_score = dict(dp_score)
                
                for g in group:
                    lag = g["lag"]
                    if lag == 1:
                        if prev_score.get(lag, -1) < g["sim"]:
                            dp_state[lag] = g
                            dp_score[lag] = g["sim"]
                    else:
                        prev_cand = prev_state.get(lag - 1)
                        if prev_cand is not None:
                            new_score = prev_score.get(lag - 1, 0.0) + g["sim"]
                            if dp_state.get(lag) is None or dp_score.get(lag, -1) < new_score:
                                dp_state[lag] = g
                                dp_score[lag] = new_score
                                trace[lag - 1][frame] = prev_cand
                
                i = j
            
            list_solutions.extend(
                self._reconstruct_sequences(vid_candidates, events, trace, top_n=10)
            )
            
            dp_state.clear()
            dp_score.clear()
            trace.clear()
        
        return self._finalise_solutions(list_solutions, events, candidates)
    
    # ── Private helpers ───────────────────────────────────────────
    
    def _gather_candidates(
        self,
        events: List[str],
        top_k: int,
        mode: str,
        option: int = 0,
    ) -> Tuple[List[List[Dict]], int]:
        """
        Fetch candidates for each event.
        
        Returns:
            Tuple of (all_candidates, special_event_index)
            special_event_index is 1-based, 0 = no starred event
        """
        all_candidates: List[List[Dict]] = []
        special_idx = 0
        
        for event_num, ev in enumerate(events, start=1):
            is_starred = ev.endswith("*")
            
            if option == 1:
                cands = self.searcher.search_text_all(ev, score=0.2, save_to_db=False)
            elif is_starred:
                special_idx = event_num
                raw = self.searcher.search_text(
                    ev[:-1], top_k=500, save_to_db=True
                )
                cands = self.reranker.rerank(ev[:-1], raw)
            else:
                cands = self.searcher.search_text(
                    ev, top_k=top_k, save_to_db=True
                )
            
            all_candidates.append(cands)
            logger.debug(f"Event {event_num}: {len(cands)} candidates")
        
        return all_candidates, special_idx
    
    def _select_candidate_videos(
        self,
        all_candidates: List[List[Dict]],
        special_idx: int,
    ) -> set:
        """Build set of videos to process."""
        if not all_candidates:
            return set()
        
        anchor_idx = (special_idx - 1) if special_idx > 0 else 0
        return {r["video_name"] for r in all_candidates[anchor_idx]}
    
    def _collect_vid_candidates(
        self,
        vid: str,
        all_event_candidates: List[List[Dict]],
    ) -> Tuple[List[Dict], bool]:
        """Collect per-event candidates for a single video."""
        flat: List[Dict] = []
        
        for lag, ev_res in enumerate(all_event_candidates, start=1):
            num_in_vid = 0
            for r in ev_res:
                if r["video_name"] != vid:
                    continue
                fi = parse_frame_int(r["frame_idx"])
                flat.append({
                    "frame": fi,
                    "image_path": r["image_path"],
                    "sim": r["similarity"],
                    "frame_str": r["frame_idx"],
                    "lag": lag,
                })
                num_in_vid += 1
            
            if num_in_vid == 0:
                return [], False
        
        return flat, True
    
    def _reconstruct_sequences(
        self,
        vid_candidates: List[Dict],
        events: List[str],
        trace: Dict[int, Dict[int, Dict]],
        top_n: int = 5,
    ) -> List[Tuple[List[Optional[Dict]], float]]:
        """Reconstruct sequences by walking trace table backwards."""
        n_events = len(events)
        candidates = []
        
        for cand in vid_candidates:
            lag = cand["lag"]
            if lag != n_events:
                continue
            if trace.get(lag - 1, {}).get(cand["frame"]) is None:
                continue
            
            sequence = [None] * n_events
            seq_score = 0.0
            cur = cand
            cur_lag = lag
            
            while cur_lag > 0:
                sequence[cur_lag - 1] = cur
                seq_score += cur["sim"]
                cur = trace.get(cur_lag - 1, {}).get(cur["frame"])
                if cur is None:
                    break
                cur_lag -= 1
            
            candidates.append((sequence, seq_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]
    
    def _finalise_solutions(
        self,
        list_solutions: List[Tuple],
        events: List[str],
        candidates: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert internal format to API output format."""
        list_solutions.sort(key=lambda x: x[1], reverse=True)
        list_solutions = list_solutions[: min(candidates, len(list_solutions))]
        
        if not list_solutions:
            logger.warning("No valid TRAKE solutions found")
            return None
        
        get_ans: List[Dict[str, Any]] = []
        
        for solution, score in list_solutions:
            if any(s is None for s in solution):
                continue
            
            frame_ids = [s["frame_str"] for s in solution]
            video_name = solution[0]["image_path"].split(os.sep)[-2]
            media_info = load_media_info(video_name)
            
            details = [
                {
                    "event": events[i],
                    "frame": solution[i]["frame_str"],
                    "sim": solution[i]["sim"],
                    "path": solution[i]["image_path"],
                }
                for i in range(len(events))
            ]
            
            get_ans.append({
                "video_name": video_name,
                "frame_ids": frame_ids,
                "details": details,
                "score": score,
                "watch_url": media_info.get("watch_url", ""),
            })
        
        logger.info(f"Found {len(get_ans)} TRAKE solutions")
        return get_ans if get_ans else None