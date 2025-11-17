#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import csv
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils import print_hl, print_error
from utils_level_wise_eval import eval_geolocation_response
from utils_geocode import extract_pred_address_v2, geocode_address, haversine_km


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _extract_final_response(pred_output: Any) -> str:
    """
    Extract the last assistant message content (string) from pred_output.
    pred_output is expected to be a list of dicts with keys: role, content.
    """
    if not isinstance(pred_output, list):
        return ""
    # scan from the end to find the last assistant message
    for i in range(len(pred_output) - 1, -1, -1):
        msg = pred_output[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        # if content is already a string, return directly
        if isinstance(content, str):
            final_text = content
            final_text = final_text.replace('<tool_call>', '')
            final_text = final_text.replace('</tool_call>', '')
            final_text = final_text.strip()
            return content[:4000]  # limit to first 4000 chars
        # if content is a list (multi-modal), extract text pieces
        if isinstance(content, list):
            texts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    t = chunk.get("text")
                    if isinstance(t, str):
                        texts.append(t)
            if texts:
                final_text = "\n".join(texts)
                final_text = final_text.replace('<tool_call>', '')
                final_text = final_text.replace('</tool_call>', '')
                final_text = final_text.strip()
                return final_text[:4000]  # limit to first 4000 chars
    return ""


def _extract_gold_location(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try to find a location dict from the record.
    Priority:
    - metadata.location
    - metadata (directly, since eval util can read many key aliases)
    - top-level 'location'
    - None
    """
    metadata = obj.get("metadata") or {}
    if isinstance(metadata, dict):
        loc = metadata.get("location")
        if isinstance(loc, dict):
            return loc
        # fall back: pass metadata itself if it has relevant keys
        if any(k in metadata for k in [
            "lat", "lng", "lon", "long", "longitude",
            "city", "country", "province_or_state", "state", "admin1", "admin2"
        ]):
            return metadata
    top_loc = obj.get("location")
    if isinstance(top_loc, dict):
        return top_loc
    return None


def _ensure_outdir(p: Optional[str]):
    if not p:
        return
    out_dir = os.path.dirname(os.path.abspath(p))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _safe_str(x: Any) -> str:
    if x is None:
        return "None"
    try:
        return str(x)
    except Exception:
        return "None"


def _default_metrics_csv_path(args: argparse.Namespace) -> Optional[str]:
    """Derive a default CSV path from out_jsonl/evaluation_jsonl/pred_jsonl."""
    base_path = args.out_jsonl or args.evaluation_jsonl or args.pred_jsonl
    if not base_path:
        return None
    d = os.path.dirname(os.path.abspath(base_path))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "eval_summary_1014.csv")


def _write_metrics_csv(
    csv_path: str,
    overall: Dict[str, Any],
    per_type_city: Dict[str, Dict[str, Any]],
    include_dist_cols: bool,
    expected_types: Optional[List[str]] = None,
) -> None:
    """
    Write a compact CSV with:
    - one 'overall' row including country/state/city accuracy and optional distance metrics
    - per data_type rows (city accuracy only)

    If expected_types is provided, ensure rows exist for those types (fill NA when absent).
    """
    header = [
        "split",
        "num_evaluated",
        "country_accuracy_pct",
        "state_accuracy_pct",
        "city_accuracy_pct",
    ]
    if include_dist_cols:
        header += ["lt_3km_pct", "median_distance_km"]

    rows: List[Dict[str, Any]] = []

    # Overall row
    ov_row: Dict[str, Any] = {
        "split": "overall",
        "num_evaluated": overall.get("num_evaluated"),
        "country_accuracy_pct": overall.get("country_accuracy_pct"),
        "state_accuracy_pct": overall.get("state_accuracy_pct"),
        "city_accuracy_pct": overall.get("city_accuracy_pct"),
    }
    if include_dist_cols:
        ov_row["lt_3km_pct"] = overall.get("lt_3km_pct", "NA")
        ov_row["median_distance_km"] = overall.get("median_distance_km", "NA")
    rows.append(ov_row)

    # Per data_type rows (city accuracy only)
    seen_types = set(per_type_city.keys())
    type_list: List[str] = []
    if expected_types:
        type_list.extend([_safe_str(t) for t in expected_types])
    # Append any extra observed types not in expected
    for t in sorted(seen_types):
        if t not in type_list:
            type_list.append(t)

    for t in type_list:
        st = per_type_city.get(t)
        if st and st.get("num"):
            city_pct = round(100.0 * st.get("city_true", 0) / max(1, st.get("num", 0)), 2)
            num = st.get("num", 0)
        else:
            city_pct = "NA"
            num = 0
        row = {
            "split": f"data_type={t}",
            "num_evaluated": num,
            "country_accuracy_pct": "NA",
            "state_accuracy_pct": "NA",
            "city_accuracy_pct": city_pct,
        }
        if include_dist_cols:
            row["lt_3km_pct"] = "NA"
            row["median_distance_km"] = "NA"
        rows.append(row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --------------------
# Shared helpers for eval reuse
# --------------------

class _EvalCaches:
    def __init__(self, dataset_dir: Optional[str]):
        self.dataset_dir = dataset_dir or ".temp/datasets/gemini-workflow-geobench-0825"
        self.ml_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self.meta_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def load_ml(self, uid: str) -> Optional[Dict[str, Any]]:
        if uid in self.ml_cache:
            return self.ml_cache[uid]
        if not uid:
            self.ml_cache[uid] = None
            return None
        ml_path = os.path.join(self.dataset_dir, uid, f"multi_level_loc_dict_{uid}.json")
        try:
            with open(ml_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.ml_cache[uid] = data
                return data
        except Exception:
            self.ml_cache[uid] = None
        return None

    def load_metadata(self, uid: str) -> Optional[Dict[str, Any]]:
        if uid in self.meta_cache:
            return self.meta_cache[uid]
        if not uid:
            self.meta_cache[uid] = None
            return None
        meta_path = os.path.join(self.dataset_dir, uid, f"metadata-{uid}.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.meta_cache[uid] = data
                return data
        except Exception:
            self.meta_cache[uid] = None
        return None

    @staticmethod
    def get_data_type_str(meta: Optional[Dict[str, Any]]) -> str:
        if not isinstance(meta, dict):
            return "None"
        val = meta.get("data_type", None)
        if val is None:
            return "None"
        if isinstance(val, str):
            v = val.strip()
            # Normalize literal "none" (any case) and empty string to canonical "None"
            if v == "" or v.lower() == "none":
                return "None"
            return v
        return _safe_str(val)


def _print_per_type_city_accuracy(per_type_city: Dict[str, Dict[str, Any]],
                                  expected_types: Optional[List[str]] = None) -> None:
    """Print per data_type City accuracy (%) in a compact JSON table style."""
    # Build ordered list of types
    seen_types = set(per_type_city.keys())
    type_list: List[str] = []
    if expected_types:
        type_list.extend([_safe_str(t) for t in expected_types])
    for t in sorted(seen_types):
        if t not in type_list:
            type_list.append(t)

    rows: List[Dict[str, Any]] = []
    for t in type_list:
        st = per_type_city.get(t)
        if st and st.get("num"):
            num = int(st.get("num", 0))
            city_true = int(st.get("city_true", 0))
            city_pct = round(100.0 * city_true / max(1, num), 2)
        else:
            num = 0
            city_true = 0
            city_pct = "NA"
        rows.append({
            "data_type": t,
            "num": num,
            "city_true": city_true,
            "city_accuracy_pct": city_pct,
        })

    print_hl("Per data_type City Accuracy (%)")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def _progress_postfix(cnt: int,
                       country_true: int,
                       state_true: int,
                       city_true: int,
                       dists: List[float],
                       show_dist: bool) -> Dict[str, str]:
    if cnt <= 0:
        return {}
    acc_country = country_true / cnt
    acc_state = state_true / cnt
    acc_city = city_true / cnt
    postfix: Dict[str, str] = {
        "acc_country": f"{acc_country:.4f}",
        "acc_state": f"{acc_state:.4f}",
        "acc_city": f"{acc_city:.4f}",
    }
    if show_dist and len(dists) >= 4:
        import math
        sorted_d = sorted(dists)
        idx50 = max(0, math.ceil(len(sorted_d) * 0.50) - 1)
        postfix["q50_km"] = f"{sorted_d[idx50]:.2f}"
    return postfix


def _finalize_and_write(args: argparse.Namespace,
                        cnt: int,
                        country_true: int,
                        state_true: int,
                        city_true: int,
                        dists: List[float],
                        per_type_city: Dict[str, Dict[str, Any]]) -> None:
    # Final accuracy
    if cnt == 0:
        print("No valid samples evaluated.")
        return

    acc_country_pct = country_true / cnt * 100
    acc_state_pct = state_true / cnt * 100
    acc_city_pct = city_true / cnt * 100

    print_hl("Final Multi-level Accuracy")
    print(json.dumps({
        "num_evaluated": cnt,
        "country_accuracy": round(acc_country_pct, 2),
        "state_accuracy": round(acc_state_pct, 2),
        "city_accuracy": round(acc_city_pct, 2),
    }, ensure_ascii=False, indent=2))

    # Distance summary (optional)
    lt3_pct_out: Optional[float] = None
    median_km_out: Optional[float] = None
    if args.eval_accurate_km_dist and dists:
        import math
        sorted_d = sorted(dists)
        avg_d = sum(sorted_d) / len(sorted_d)
        idx25 = max(0, math.ceil(len(sorted_d) * 0.25) - 1)
        idx50 = max(0, math.ceil(len(sorted_d) * 0.50) - 1)
        idx75 = max(0, math.ceil(len(sorted_d) * 0.75) - 1)
        # extra metrics
        lt3 = sum(1 for x in sorted_d if x < 3.0)
        lt3_pct = round(100.0 * lt3 / max(1, len(sorted_d)), 2)
        median_km = round(sorted_d[idx50], 4)
        lt3_pct_out = lt3_pct
        median_km_out = float(median_km)

        print_hl("Accurate Distance Summary (km)")
        print(json.dumps({
            "num_dist": len(sorted_d),
            "avg_dist_km": round(avg_d, 2),
            "q25_nearest_km": round(sorted_d[idx25], 2),
            "q50_nearest_km": round(sorted_d[idx50], 2),
            "q75_nearest_km": round(sorted_d[idx75], 2),
        }, ensure_ascii=False, indent=2))

        # Additional metrics unless suppressed
        if not args.no_eval_accurate_dist:
            print_hl("Additional Distance Metrics")
            print(json.dumps({
                "lt_3km_pct": lt3_pct,
                "median_distance_km": median_km,
            }, ensure_ascii=False, indent=2))

    # Prepare and write CSV
    csv_path = args.metrics_csv or _default_metrics_csv_path(args)
    # Prepare expected types list once for both printing and CSV
    expected_types: Optional[List[str]] = None
    if args.data_types:
        expected_types = [s.strip() for s in args.data_types.split(",") if s.strip()]
        expected_types = [("None" if s.lower() == "none" else s) for s in expected_types]
    else:
        # Default to showing a placeholder row for data_type=None
        expected_types = ["None"]

    # Print per data_type city accuracy (%)
    _print_per_type_city_accuracy(per_type_city, expected_types)

    if csv_path:
        overall_row = {
            "num_evaluated": cnt,
            "country_accuracy_pct": round(acc_country_pct, 2),
            "state_accuracy_pct": round(acc_state_pct, 2),
            "city_accuracy_pct": round(acc_city_pct, 2),
        }
        include_dist_cols = False
        if args.eval_accurate_km_dist and (not args.no_eval_accurate_dist):
            include_dist_cols = True
            if dists:
                overall_row["lt_3km_pct"] = round(100.0 * sum(1 for x in dists if x < 3.0) / max(1, len(dists)), 2)
                import math
                sorted_d2 = sorted(dists)
                idx50_2 = max(0, math.ceil(len(sorted_d2) * 0.50) - 1)
                overall_row["median_distance_km"] = float(round(sorted_d2[idx50_2], 2))
            else:
                overall_row["lt_3km_pct"] = "NA"
                overall_row["median_distance_km"] = "NA"

        _write_metrics_csv(
            csv_path=csv_path,
            overall=overall_row,
            per_type_city=per_type_city,
            include_dist_cols=include_dist_cols,
            expected_types=expected_types,
        )
        if args.debug:
            print_hl(f"Metrics CSV written: {csv_path}")


def _eval_single(obj: Dict[str, Any], args: argparse.Namespace, caches: _EvalCaches) -> Dict[str, Any]:
    """Evaluate one record and return a standard dict.

    Return shape on success:
    {
      "uid", "image_path", "success": True,
      "row": {
        uid, image_path, country_correct, state_correct, city_correct, metadata_data_type,
        [extract_address, geocode_result, accurate_km_dist]
      },
      "debug": {response_text, loc_dict, result, extract_addr, geocode_res, dist_km}
    }

    On failure, returns:
    {"uid", "image_path", "success": False, "error": str, "metadata_data_type": dt, "debug": {response_text}}
    """
    uuid = obj.get("uid") or obj.get("uuid") or ""
    image_path = obj.get("image_path")

    response_text = _extract_final_response(obj.get("pred_output"))

    loc_dict = _extract_gold_location(obj) or {}
    ml = caches.load_ml(uuid) or loc_dict["multi_level_loc_dict"]
    if isinstance(ml, dict):
        loc_dict["multi_level_loc_dict"] = ml

    # Default return payload
    ret: Dict[str, Any] = {
        "uid": uuid,
        "image_path": image_path,
        "success": False,
        "debug": {"response_text": response_text},
    }

    # Preload metadata for data_type and possible GT coords
    meta = caches.load_metadata(uuid) or obj.get("metadata", {})
    dt = caches.get_data_type_str(meta)

    try:
        if response_text == "":
            raise ValueError(f"expect response for {uuid}")
        result = eval_geolocation_response(
            response=response_text,
            loc_dict=loc_dict['multi_level_loc_dict'],
            model_verifier=args.model_verifier,
            api_key=args.api_key,
            timeout=args.timeout,
            debug_mode=args.debug,
        )
    except Exception as e:
        ret.update({"error": str(e), "metadata_data_type": dt})
        return ret

    # Optional geocoding
    extract_addr: Optional[str] = None
    geocode_res: Optional[Dict[str, Any]] = None
    dist_km: Optional[float] = None
    if args.eval_accurate_km_dist:
        try:
            gt_loc = meta.get("location") if isinstance(meta, dict) else None
            if isinstance(gt_loc, dict) and not (dt == "planetary"):
                extract_addr = extract_pred_address_v2(
                    response_text, api_key=args.api_key, timeout=args.timeout, debug=args.debug
                )
                start_ts = time.perf_counter()
                geocode_res = geocode_address(
                    extract_addr or "", timeout=args.timeout, debug=True
                )
                geocode_ms = (time.perf_counter() - start_ts) * 1000.0
                dist_km = haversine_km(geocode_res, gt_loc)
        except Exception as e:
            ret.update({"error": str(e), "metadata_data_type": dt})
            return ret

    # Build output row
    row = {
        "uid": uuid,
        "image_path": image_path,
        "country_correct": bool(result.get("country_correct")),
        "state_correct": bool(result.get("state_correct")),
        "city_correct": bool(result.get("city_correct")),
        "metadata_data_type": dt,
    }
    if args.eval_accurate_km_dist:
        row.update({
            "extract_address": extract_addr,
            "geocode_result": geocode_res,
            "accurate_km_dist": dist_km,
        })

    ret.update({
        "success": True,
        "row": row,
        "debug": {
            "response_text": response_text,
            "loc_dict": loc_dict,
            "result": result,
            "extract_addr": extract_addr,
            "geocode_res": geocode_res,
            "dist_km": dist_km,
            "geocode_runtime_ms": geocode_ms if args.eval_accurate_km_dist and ("geocode_ms" in locals()) else None,
        },
    })
    return ret


def _sequential_eval(args: argparse.Namespace) -> None:
    # Load input
    items = _read_jsonl(args.pred_jsonl)
    if args.num_samples:
        items = items[:args.num_samples]

    total = len(items)
    if total == 0:
        print(f"No items to evaluate in: {args.pred_jsonl}")
        return

    if args.debug:
        print_hl("Args")
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        print_hl("Loaded items")
        print(total)

    _ensure_outdir(args.out_jsonl)

    # Accumulators
    cnt = 0
    country_true = 0
    state_true = 0
    city_true = 0
    # Distances (only used when --eval_accurate_km_dist)
    dists: List[float] = []

    # Per data_type city stats
    per_type_city: Dict[str, Dict[str, int]] = {}

    out_fp = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    # shared caches/context
    caches = _EvalCaches(args.dataset_dir)

    with tqdm(total=total, desc="Evaluating") as pbar:
        for obj in items:
            uuid = obj.get("uid") or obj.get("uuid") or ""
            image_path = obj.get("image_path")

            # Pre-eval debug (model response)
            if args.debug:
                response_text = _extract_final_response(obj.get("pred_output"))
                print_hl(f"Sample {uuid or '[no-uid]'}")
                print_hl("Model response")
                print((response_text or "")[:2000])

            ret = _eval_single(obj, args, caches)

            if not ret.get("success"):
                if args.debug:
                    print_error(f"Sample {uuid or '[no-uid]'} - ERROR")
                    print(_safe_str(ret.get("error")))
                if out_fp:
                    err_row = {
                        "uid": uuid,
                        "image_path": image_path,
                        "error": ret.get("error"),
                    }
                    out_fp.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                pbar.update(1)
                continue

            row = ret.get("row", {})
            debug_info = ret.get("debug", {})

            # Update counters
            cnt += 1
            if row.get("country_correct"):
                country_true += 1
            if row.get("state_correct"):
                state_true += 1
            if row.get("city_correct"):
                city_true += 1

            # Per data_type city stats
            dt = _safe_str(row.get("metadata_data_type"))
            entry = per_type_city.setdefault(dt, {"num": 0, "city_true": 0})
            entry["num"] += 1
            if row.get("city_correct"):
                entry["city_true"] += 1

            # Distances
            if args.eval_accurate_km_dist and (row.get("accurate_km_dist") is not None):
                try:
                    dists.append(float(row.get("accurate_km_dist")))
                except (TypeError, ValueError):
                    pass

            # Update progress bar
            postfix = _progress_postfix(cnt, country_true, state_true, city_true, dists, args.eval_accurate_km_dist)
            if postfix:
                pbar.set_postfix(postfix)

            # Debug printing (detailed)
            if args.debug:
                loc_dict = debug_info.get("loc_dict")
                print_hl("Gold location (used)")
                try:
                    print(json.dumps(loc_dict, ensure_ascii=False, indent=2))
                except Exception:
                    print(_safe_str(loc_dict))
                print_hl("Eval result (booleans)")
                print(json.dumps(debug_info.get("result"), ensure_ascii=False, indent=2))
                if args.eval_accurate_km_dist:
                    geocode_ms = debug_info.get("geocode_runtime_ms")
                    if geocode_ms is not None:
                        print_hl(f"geocode_address runtime: {float(geocode_ms):.2f} ms")
                    print_hl("Extraction / Geocode / Distance")
                    print(json.dumps({
                        "extract_address": debug_info.get("extract_addr"),
                        "geocode_result": debug_info.get("geocode_res"),
                        "accurate_km_dist": debug_info.get("dist_km"),
                    }, ensure_ascii=False, indent=2))

            # Write per-sample evaluation output if requested
            if out_fp:
                out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

            pbar.update(1)

    if out_fp:
        out_fp.close()

    _finalize_and_write(args, cnt, country_true, state_true, city_true, dists, per_type_city)


def _concurrent_eval(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Load input
    items = _read_jsonl(args.pred_jsonl)
    if args.num_samples:
        items = items[:args.num_samples]

    total = len(items)
    if total == 0:
        print(f"No items to evaluate in: {args.pred_jsonl}")
        return

    if args.debug:
        print_hl("Args")
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        print_hl("Loaded items")
        print(total)

    _ensure_outdir(args.out_jsonl)

    out_fp = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    # shared caches/context
    caches = _EvalCaches(args.dataset_dir)

    # Per-type stats and global accumulators
    cnt = 0
    country_true = 0
    state_true = 0
    city_true = 0
    dists: List[float] = []
    per_type_city: Dict[str, Dict[str, int]] = {}

    # Result buffering to keep order when writing
    next_write = 0
    buffer: Dict[int, Dict[str, Any]] = {}

    def _process_one(idx: int, obj: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        uuid = obj.get("uid") or obj.get("uuid") or ""
        if args.debug:
            try:
                response_text = _extract_final_response(obj.get("pred_output"))
                print_hl(f"Sample {uuid or '[no-uid]'}")
                print_hl("Model response")
                print((response_text or "")[:2000])
            except Exception:
                pass
        ret = _eval_single(obj, args, caches)
        return idx, ret

    with tqdm(total=total, desc="Evaluating") as pbar:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futures = [ex.submit(_process_one, i, items[i]) for i in range(total)]
            for fut in as_completed(futures):
                idx, ret = fut.result()
                buffer[idx] = ret
                # Stream ordered writes and stats updates
                while next_write in buffer:
                    cur = buffer.pop(next_write)
                    # Update progress/statistics
                    if cur.get("success"):
                        row = cur.get("row") or {}
                        cnt += 1
                        if row.get("country_correct"): country_true += 1
                        if row.get("state_correct"): state_true += 1
                        if row.get("city_correct"): city_true += 1
                        dt = _safe_str(row.get("metadata_data_type"))
                        entry = per_type_city.setdefault(dt, {"num": 0, "city_true": 0})
                        entry["num"] += 1
                        if row.get("city_correct"): entry["city_true"] += 1
                        if args.eval_accurate_km_dist and (row.get("accurate_km_dist") is not None):
                            try:
                                dists.append(float(row.get("accurate_km_dist")))
                            except (TypeError, ValueError):
                                pass
                        # Write out in order
                        if out_fp:
                            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:
                        # write error row if needed (keeps previous behavior)
                        if out_fp:
                            out_fp.write(json.dumps(cur, ensure_ascii=False) + "\n")
                    next_write += 1
                    pbar.update(1)

                    postfix = _progress_postfix(cnt, country_true, state_true, city_true, dists, args.eval_accurate_km_dist)
                    if postfix:
                        pbar.set_postfix(postfix)

    if out_fp:
        out_fp.close()

    _finalize_and_write(args, cnt, country_true, state_true, city_true, dists, per_type_city)


def _only_eval_aggregate(args: argparse.Namespace) -> None:
    """
    Fast path that aggregates an already-produced evaluation JSONL
    and prints the same final metrics. Also writes the CSV summary.
    """
    items = _read_jsonl(args.evaluation_jsonl)
    if args.num_samples:
        items = items[:args.num_samples]
    total = len(items)
    if total == 0:
        print(f"No items to evaluate in: {args.evaluation_jsonl}")
        return

    # Aggregate booleans; skip rows without the boolean keys (e.g., error rows)
    dists: List[float] = []
    cnt = 0
    country_true = 0
    state_true = 0
    city_true = 0

    # Per data_type city stats
    per_type_city: Dict[str, Dict[str, int]] = {}

    # For per-type stats we need metadata; use dataset_dir
    dataset_dir = args.dataset_dir or ".temp/datasets/gemini-workflow-geobench-0825"

    def _load_metadata(uid: str) -> Optional[Dict[str, Any]]:
        meta_path = os.path.join(dataset_dir, uid, f"metadata-{uid}.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _get_data_type_str(meta: Optional[Dict[str, Any]]) -> str:
        if not isinstance(meta, dict):
            return "None"
        val = meta.get("data_type")
        if val is None:
            return "None"
        if isinstance(val, str):
            v = val.strip()
            # Normalize literal "none" (any case) and empty string to canonical "None"
            if v == "" or v.lower() == "none":
                return "None"
            return v
        return _safe_str(val)

    for row in items:
        if not isinstance(row, dict):
            continue
        dist_val = row.get("accurate_km_dist")
        if dist_val is not None:
            try:
                dists.append(float(dist_val))
            except (TypeError, ValueError):
                pass
        if ("country_correct" in row) and ("state_correct" in row) and ("city_correct" in row):
            cnt += 1
            if bool(row.get("country_correct")): country_true += 1
            if bool(row.get("state_correct")): state_true += 1
            if bool(row.get("city_correct")): city_true += 1

            # per-type
            uid = row.get("uid") or row.get("uuid") or ""
            meta = _load_metadata(uid) if uid else None
            dt = _get_data_type_str(meta)
            entry = per_type_city.setdefault(dt, {"num": 0, "city_true": 0})
            entry["num"] += 1
            if bool(row.get("city_correct")):
                entry["city_true"] += 1

    if cnt == 0:
        print("No valid samples evaluated.")
        return

    acc_country = country_true / cnt
    acc_state = state_true / cnt
    acc_city = city_true / cnt

    print_hl("Final Multi-level Accuracy")
    print(json.dumps({
        "num_evaluated": cnt,
        "country_accuracy": round(acc_country, 4),
        "state_accuracy": round(acc_state, 4),
        "city_accuracy": round(acc_city, 4),
    }, ensure_ascii=False, indent=2))

    # Print per data_type City (%) as well
    expected_types: Optional[List[str]] = None
    if args.data_types:
        expected_types = [s.strip() for s in args.data_types.split(",") if s.strip()]
        expected_types = [("None" if s.lower() == "none" else s) for s in expected_types]
    else:
        # By default, include a placeholder row for data_type=None
        expected_types = ["None"]
    _print_per_type_city_accuracy(per_type_city, expected_types)

    lt3_pct_out: Optional[float] = None
    median_km_out: Optional[float] = None
    if dists:
        import math
        sorted_d = sorted(dists)
        avg_d = sum(sorted_d) / len(sorted_d)
        idx25 = max(0, math.ceil(len(sorted_d) * 0.25) - 1)
        idx50 = max(0, math.ceil(len(sorted_d) * 0.50) - 1)
        idx75 = max(0, math.ceil(len(sorted_d) * 0.75) - 1)
        print_hl("Accurate Distance Summary (km)")
        print(json.dumps({
            "num_dist": len(sorted_d),
            "avg_dist_km": round(avg_d, 2),
            "q25_nearest_km": round(sorted_d[idx25], 2),
            "q50_nearest_km": round(sorted_d[idx50], 2),
            "q75_nearest_km": round(sorted_d[idx75], 2),
        }, ensure_ascii=False, indent=2))
        # Additional metrics unless suppressed
        lt3 = sum(1 for x in sorted_d if x < 3.0)
        lt3_pct = round(100.0 * lt3 / max(1, len(sorted_d)), 2)
        median_km = round(sorted_d[idx50], 2)
        lt3_pct_out = lt3_pct
        median_km_out = float(median_km)
        if not args.no_eval_accurate_dist:
            print_hl("Additional Distance Metrics")
            print(json.dumps({
                "lt_3km_pct": lt3_pct,
                "median_distance_km": median_km,
            }, ensure_ascii=False, indent=2))

    # CSV output
    csv_path = args.metrics_csv or _default_metrics_csv_path(args)
    if csv_path:
        overall_row = {
            "num_evaluated": cnt,
            "country_accuracy_pct": round(acc_country * 100, 2),
            "state_accuracy_pct": round(acc_state * 100, 2),
            "city_accuracy_pct": round(acc_city * 100, 2),
        }
        include_dist_cols = False
        if not args.no_eval_accurate_dist and dists:
            include_dist_cols = True
            overall_row["lt_3km_pct"] = lt3_pct_out if lt3_pct_out is not None else "NA"
            overall_row["median_distance_km"] = median_km_out if median_km_out is not None else "NA"

        expected_types: Optional[List[str]] = None
        if args.data_types:
            expected_types = [s.strip() for s in args.data_types.split(",") if s.strip()]
            expected_types = [("None" if s.lower() == "none" else s) for s in expected_types]
        else:
            # By default, include a placeholder row for data_type=None
            expected_types = ["None"]

        _write_metrics_csv(
            csv_path=csv_path,
            overall=overall_row,
            per_type_city=per_type_city,
            include_dist_cols=include_dist_cols,
            expected_types=expected_types,
        )
        if args.debug:
            print_hl(f"Metrics CSV written: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate geolocation predictions and output CSV summary.")
    parser.add_argument("--pred_jsonl", type=str, default=None, help="Path to inference results JSONL.")
    parser.add_argument("--out_jsonl", type=str, default=None, help="Optional: path to save per-sample evaluation results.")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--model_verifier", action="store_true", help="Enable model-based verifier inside eval.")
    parser.add_argument("--api_key", type=str, default=None, help="API key if model_verifier enabled.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout seconds for model_verifier.")
    parser.add_argument("--debug", action="store_true", help="Verbose debug printing.")
    parser.add_argument("--dataset_dir", type=str, default=".temp/datasets/gemini-workflow-geobench-0825", help="Dataset root to read multi_level_loc_dict_<uid>.json and metadata-<uid>.json")
    parser.add_argument("--only_eval", action="store_true", help="Only aggregate metrics from an evaluation JSONL (no re-evaluation).")
    parser.add_argument("--evaluation_jsonl", type=str, default=None, help="Path to evaluation JSONL when --only_eval is set.")
    parser.add_argument("--eval_accurate_km_dist", action="store_true", help="Also compute accurate km distance via address extraction + geocoding against GT metadata.")
    parser.add_argument("--no_eval_accurate_dist", action="store_true", help="When set, do not output the two additional distance metrics (<3km %, median km) even if distances are computed.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers. 1 = sequential (identical to original).")
    parser.add_argument("--metrics_csv", type=str, default=None, help="Optional path to write CSV summary. Defaults next to out_jsonl/evaluation_jsonl/pred_jsonl.")
    parser.add_argument("--data_types", type=str, default=None, help="Optional comma-separated list of expected data_type values for CSV rows; missing ones will be filled as NA.")

    args = parser.parse_args()
    args.eval_accurate_km_dist = not args.no_eval_accurate_dist
    print(f'{args.eval_accurate_km_dist=}')

    # Only-eval fast path (sequential aggregation)
    if args.only_eval:
        if not args.evaluation_jsonl:
            print("--evaluation_jsonl is required when --only_eval is set.")
            return
        return _only_eval_aggregate(args)

    # Choose codepath
    if int(args.workers) <= 1:
        # Fully sequential, consistent behavior
        return _sequential_eval(args)
    else:
        return _concurrent_eval(args)


if __name__ == "__main__":
    main()
