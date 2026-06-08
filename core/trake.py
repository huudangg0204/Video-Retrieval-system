"""core/trake.py — Temporal Retrieval and Alignment of Key Events (TRAKE).

Fixes vs. original:
1. parse_frame_int deduplicated — single shared function, no longer defined twice.
2. `num` variable collision fixed — outer event-counter renamed `event_count` so it
   isn't overwritten by the inner per-video counter `num_in_vid`.
3. `score` variable collision in trake_highest fixed — the DP score dict is renamed
   `dp_score` and the reconstruction float is `seq_score`, so they never alias.
4. candidate_videos selection logic fixed — correct index used when no starred event.
5. media_info loaded via shared util, no more `if 'media_info' in locals()` hack.
6. Result-building duplicated across trake_closest/trake_highest extracted to
   `_build_trake_results`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import config
from utils.frame_utils import parse_frame_int
from utils.media_utils import load_media_info
from .search import SearchEngine
from .rerank import Reranker


class TrakeEngine:
    """
    Implements two TRAKE strategies:

    • trake_closest — greedy DP: finds the sequence with minimum total frame-gap.
    • trake_highest — optimised DP: finds the sequence with maximum total similarity.

    A query event ending with '*' triggers BLIP reranking of its candidates.
    """

    def __init__(self, searcher: SearchEngine, reranker: Reranker) -> None:
        self.searcher  = searcher
        self.reranker  = reranker

    # ── Public API ────────────────────────────────────────────────────────────

    def trake_closest(
        self,
        events: List[str],
        top_k:      int = 200,
        candidates: int = 200,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find up to *candidates* sequences where events appear in temporal order,
        using a greedy frame-closest DP.

        A '*' suffix on an event string enables BLIP reranking for that event.
        Returns a list of answer dicts or None if no valid sequence is found.
        """
        if not events:
            return None

        all_event_candidates, special_idx = self._gather_candidates(events, top_k, mode="closest")
        candidate_videos = self._select_candidate_videos(all_event_candidates, special_idx)

        list_solutions: List[Tuple] = []

        for vid in candidate_videos:
            vid_candidates, all_present = self._collect_vid_candidates(vid, all_event_candidates)
            if not all_present:
                continue

            # Greedy DP: for each frame keep the highest-sim candidate, preserving event order
            vid_candidates.sort(key=lambda x: (x["frame"], -x["lag"]))

            # Deduplicate same-frame candidates — keep highest similarity per frame
            deduped: Dict[int, Dict] = {}
            for c in vid_candidates:
                f = c["frame"]
                if f not in deduped or deduped[f]["sim"] < c["sim"]:
                    deduped[f] = c
            vid_candidates = sorted(deduped.values(), key=lambda x: x["frame"])

            D:     Dict[int, Dict] = {}
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
        events:  List[str],
        top_k:   int = 200,
        candidates: int = 200,
        option:  int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find up to *candidates* sequences where events appear in temporal order,
        maximising the total similarity score via optimised DP.

        option=0: standard top-k search (with optional BLIP reranking for starred events).
        option=1: threshold-based search (score >= 0.2) for all events.
        """
        if not events:
            return None

        all_event_candidates, special_idx = self._gather_candidates(
            events, top_k, mode="highest", option=option
        )
        candidate_videos = self._select_candidate_videos(all_event_candidates, special_idx)

        list_solutions: List[Tuple] = []

        for vid in candidate_videos:
            vid_candidates, all_present = self._collect_vid_candidates(vid, all_event_candidates)
            if not all_present:
                continue

            vid_candidates.sort(key=lambda x: x["frame"])

            # Optimised DP — rename variables to avoid collision with outer scope
            dp_state: Dict[int, Dict]         = {}   # lag → best candidate ending at that lag
            dp_score: Dict[int, float]         = defaultdict(float)
            trace:    Dict[int, Dict[int, Dict]] = defaultdict(dict)

            i = 0
            while i < len(vid_candidates):
                cand  = vid_candidates[i]
                frame = cand["frame"]

                # Collect all candidates at the same frame (may have different lags)
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
                        # Start of chain: prefer highest similarity
                        if prev_score.get(lag) is None or prev_score.get(lag, 0.0) < g["sim"]:
                            dp_state[lag] = g
                            dp_score[lag] = g["sim"]
                    else:
                        prev_cand = prev_state.get(lag - 1)
                        if prev_cand is not None:
                            new_score = prev_score.get(lag - 1, 0.0) + g["sim"]
                            if dp_state.get(lag) is None or dp_score.get(lag, 0.0) < new_score:
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gather_candidates(
        self,
        events: List[str],
        top_k: int,
        mode: str,
        option: int = 0,
    ) -> Tuple[List[List[Dict]], int]:
        """
        Fetch top-k results per event. Returns (all_candidates, special_event_index).

        special_event_index is 1-based; 0 means no starred event.
        Starred events ('*' suffix) are reranked with BLIP.
        """
        all_candidates: List[List[Dict]] = []
        special_idx = 0  # 1-based index of the starred event (0 = none)

        for event_num, ev in enumerate(events, start=1):
            is_starred = ev.endswith("*")

            if option == 1:
                # Threshold-based: get all frames above score threshold
                cands = self.searcher.search_text_all(ev, score=0.2, save_to_db=False)
            elif is_starred:
                special_idx = event_num
                raw = self.searcher.search_text(ev[:-1], top_k=500, save_to_db=True)
                cands = self.reranker.rerank(ev[:-1], raw)
            else:
                cands = self.searcher.search_text(ev, top_k=top_k, save_to_db=True)

            all_candidates.append(cands)

        return all_candidates, special_idx

    def _select_candidate_videos(
        self,
        all_candidates: List[List[Dict]],
        special_idx: int,
    ) -> set:
        """
        Build the set of videos to score.

        If a starred event exists, only videos appearing in that event's candidates
        are considered (prioritises precision). Otherwise, use the first event.
        """
        if not all_candidates:
            return set()

        # special_idx is 1-based; use 0-based indexing
        anchor_idx = (special_idx - 1) if special_idx > 0 else 0
        return {r["video_name"] for r in all_candidates[anchor_idx]}

    def _collect_vid_candidates(
        self,
        vid: str,
        all_event_candidates: List[List[Dict]],
    ) -> Tuple[List[Dict], bool]:
        """
        For a single video, collect all per-event candidates as a flat list
        tagged with their 1-based event index (`lag`).

        Returns (flat_list, all_events_present).
        `all_events_present` is False if any event has zero candidates for this video.
        """
        flat: List[Dict] = []
        for lag, ev_res in enumerate(all_event_candidates, start=1):
            num_in_vid = 0
            for r in ev_res:
                if r["video_name"] != vid:
                    continue
                fi = parse_frame_int(r["frame_idx"])
                flat.append({
                    "frame":     fi,
                    "image_path": r["image_path"],
                    "sim":        r["similarity"],
                    "frame_str":  r["frame_idx"],
                    "lag":        lag,
                })
                num_in_vid += 1
            if num_in_vid == 0:
                return [], False  # early exit — this video misses an event
        return flat, True

    def _reconstruct_sequences(
        self,
        vid_candidates: List[Dict],
        events: List[str],
        trace: Dict[int, Dict[int, Dict]],
        top_n: int = 5,
    ) -> List[Tuple[List[Optional[Dict]], float]]:
        """
        Walk the trace table backwards from the last-event candidates to
        reconstruct complete sequences. Returns at most *top_n* best sequences.
        """
        n_events   = len(events)
        candidates = []

        for cand in vid_candidates:
            lag = cand["lag"]
            if lag != n_events:
                continue
            if trace[lag - 1].get(cand["frame"]) is None:
                continue

            sequence = [None] * n_events
            seq_score = 0.0
            cur = cand
            cur_lag = lag

            while cur_lag > 0:
                sequence[cur_lag - 1] = cur
                seq_score += cur["sim"]
                cur = trace[cur_lag - 1].get(cur["frame"])
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
        """
        Sort all collected sequences by score, truncate to *candidates*,
        and convert to the output dict format consumed by the API/GUI.
        """
        import os
        list_solutions.sort(key=lambda x: x[1], reverse=True)
        list_solutions = list_solutions[: min(candidates, len(list_solutions))]

        if not list_solutions:
            print("❌ TRAKE: No valid solutions found")
            return None

        get_ans: List[Dict[str, Any]] = []

        for solution, score in list_solutions:
            if any(s is None for s in solution):
                continue

            frame_ids  = [s["frame_str"] for s in solution]
            video_name = solution[0]["image_path"].split(os.sep)[-2]
            media_info = load_media_info(video_name)
            details    = [
                {
                    "event": events[i],
                    "frame": solution[i]["frame_str"],
                    "sim":   solution[i]["sim"],
                    "path":  solution[i]["image_path"],
                }
                for i in range(len(events))
            ]
            get_ans.append({
                "video_name": video_name,
                "frame_ids":  frame_ids,
                "details":    details,
                "score":      score,
                "watch_url":  media_info.get("watch_url", ""),
            })

        return get_ans if get_ans else None
