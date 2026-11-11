import hashlib
import json
import re
from pathlib import Path

import config


def split_by_headings(text: str):
    """按 # ~ #### 标题行切块，返回 [(heading, body), ...]"""
    lines = text.split("\n")   # ① 按换行符拆成一行行
    blocks = []                # ② 结果列表
    cur_heading = ""           # ③ 当前标题
    cur_lines = []             # ④ 当前正文（正在攒的一堆行）

    for line in lines:
        if re.match(r"^#{1,4}\s", line):      # ⑤ 这行是标题吗？
            if cur_lines:                      # ⑥ 手里攒着正文，就先把上一块收尾
                blocks.append((cur_heading, "\n".join(cur_lines)))
            cur_heading = line.strip().lstrip("#").strip()  # ⑦ 记新标题，去掉 #
            cur_lines = []                     # ⑧ 清空，开始攒新块
        else:
            cur_lines.append(line)             # ⑨ 不是标题，塞进当前正文

    if cur_lines:                              # ⑩ 最后一块收尾
        blocks.append((cur_heading, "\n".join(cur_lines)))
    return blocks

def clean_body(body):
    """清洗：去图片、去纯链接行、压空行"""
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)","",body)
    kept = []
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r"(\[[^\]]*\]\([^)]*\)[\s，,、；;]*)+",s):
            continue
        kept.append(s)
    return "\n".join(kept)

def pack_paragraphs(paragraphs,max_chars):
    """段落贪心打包成 ≤ max_chars 的块，块间保留 overlap"""
    chunks = []
    cur = []
    cur_len = 0
    for p in paragraphs:
        if cur and (cur_len + len(p) > max_chars):
            chunks.append("\n".join(cur))
            cur = [cur[-1]]
            cur_len = len(cur[-1])
        cur.append(p)
        cur_len += len(p)
    if cur:
        chunks.append("\n".join(cur))
    return chunks

def chunk_file(path: Path, seen_hashes: set):
    """单文件 → chunk 列表"""
    text = path.read_text(encoding="utf-8")
    chunks = []
    for heading, body in split_by_headings(text):
        if "目录" in heading and len(body) < 200:   # 跳过目录块
            continue
        body = clean_body(body)
        if not body.strip():
            continue
        paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
        for piece in pack_paragraphs(paragraphs, config.CHUNK_MAX_CHARS):
            full = f"{heading}\n{piece}" if heading else piece
            h = hashlib.md5(full.encode("utf-8")).hexdigest()
            if h in seen_hashes:                    # 去重
                continue
            seen_hashes.add(h)
            chunks.append({"source": path.name, "heading": heading, "text": full})
    return chunks


def main():
    files = sorted(config.RAW_DIR.glob("*.md"))
    print(f"共发现 {len(files)} 个 md 文件")
    all_chunks = []
    seen = set()
    for f in files:
        cs = chunk_file(f, seen)
        all_chunks.extend(cs)
        print(f"  {f.name}: {len(cs)} chunks")

    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as out:
        for i, c in enumerate(all_chunks):
            c["id"] = i
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n共生成 {len(all_chunks)} 个 chunk，已写入 {config.CHUNKS_FILE.name}")

if __name__ == "__main__":
    main()
