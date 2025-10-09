#!/usr/bin/env python3
import json, os, time, urllib.parse, urllib.request
from pathlib import Path

API = "https://happytreefriends.fandom.com/api.php"

def api_get(params, sleep=0.25):
    params = {**params, "format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, headers={"User-Agent": "gallery-fetcher/3.0 (+local)"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    if sleep:
        time.sleep(sleep)
    return json.loads(data.decode("utf-8"))

def list_category_pages(category_title):
    cont = {}
    while True:
        resp = api_get({
            "action": "query", "list": "categorymembers",
            "cmtitle": category_title, "cmtype": "page", "cmlimit": "max", **cont
        })
        for it in resp.get("query", {}).get("categorymembers", []):
            yield it["title"]
        cont = resp.get("continue") or {}
        if not cont:
            break

def images_for_page(page_title):
    cont = {}
    while True:
        resp = api_get({
            "action": "query", "generator": "images",
            "titles": page_title, "gimlimit": "max",
            "prop": "imageinfo",
            "iiprop": "url|mime|size|sha1",
            **cont,
        })
        for p in resp.get("query", {}).get("pages", []):
            if p.get("missing"): 
                continue
            ii = p.get("imageinfo") or []
            if not ii: 
                continue
            info = ii[0]
            yield {
                "filetitle": p["title"],            # "File:Something.png"
                "url": info.get("url"),
                "sha1": (info.get("sha1") or "").lower(),
                "mime": info.get("mime")            # e.g. "image/png"
            }
        cont = resp.get("continue") or {}
        if not cont:
            break

def ext_from_mime(mime):
    if not mime: return ""
    m = mime.lower()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/tiff": ".tif",
        "image/bmp": ".bmp",
        "image/svg+xml": ".svg",
    }.get(m, "")

def filename_from_filetitle(filetitle, mime):
    # strip "File:" and sanitize path separators
    base = filetitle.split(":", 1)[-1].replace("/", "_").replace("\\", "_")
    stem, ext = os.path.splitext(base)
    # if the title lacks/has weird extension, enforce from MIME
    enforced = ext_from_mime(mime)
    if enforced and enforced.lower() != ext.lower():
        return stem + enforced
    return base

def choose_target(out_dir: Path, desired: str, sha1: str, name_map: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / desired
    if desired in name_map:
        if name_map[desired] == sha1:
            return path, True   # same content already saved with this name
        # different content with same name — disambiguate
        stem, ext = os.path.splitext(desired)
        return out_dir / f"{stem}-{sha1[:8]}{ext}", False
    name_map[desired] = sha1
    if path.exists():
        # conservative fallback if file already exists on disk
        stem, ext = os.path.splitext(desired)
        i = 1
        while True:
            cand = out_dir / f"{stem} ({i}){ext}"
            if not cand.exists():
                return cand, False
            i += 1
    return path, False

def download(url, target: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "gallery-fetcher/3.0 (+local)"})
    with urllib.request.urlopen(req) as r, open(target, "wb") as f:
        f.write(r.read())

def main():
    category = "Category:Image_Galleries"
    out_dir = Path("data")

    pages = list(list_category_pages(category))
    print(f"Found {len(pages)} pages.")

    seen_sha1 = set()
    name_map = {}
    saved = skipped = 0

    for i, title in enumerate(pages, 1):
        print(f"[{i}/{len(pages)}] {title}")
        for it in images_for_page(title):
            sha1 = it["sha1"]
            if sha1 and sha1 in seen_sha1:
                skipped += 1
                continue
            desired = filename_from_filetitle(it["filetitle"], it["mime"])
            target, do_skip = choose_target(out_dir, desired, sha1, name_map)
            if do_skip:
                skipped += 1
                continue
            try:
                download(it["url"], target)
                saved += 1
                if sha1: seen_sha1.add(sha1)
                print(f"  ✓ {target.name}")
            except Exception as e:
                print(f"  ✗ {desired}: {e}")
        time.sleep(0.2)

    print(f"\nDone. Saved {saved} files to {out_dir.resolve()} (skipped {skipped} duplicates).")

if __name__ == "__main__":
    main()