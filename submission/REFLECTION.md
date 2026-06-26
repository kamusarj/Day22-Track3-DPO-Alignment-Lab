# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Bùi Hoàng Linh
**Cohort:** A20-K1
**MSSV:** 2A202600804
**Tier đã chạy:** T4 (custom) — Qwen2.5-1.5B-bnb-4bit trên RTX 3050 6GB
**Date:** 2026-06-26

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | RTX 3050 6GB Laptop GPU |
| CUDA / driver | CUDA 13.0, driver 580.159 |
| Base model | unsloth/Qwen2.5-1.5B-bnb-4bit |
| SFT dataset slice | tatsu-lab/alpaca · 300 samples · 1 epoch |
| Preference dataset slice | Synthetic từ Alpaca · 247 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 (custom override: BASE_MODEL, MAX_LEN=384) |
| Total cost | $0 (local GPU) |

> Ghi chú: Vì GPU chỉ có 6 GB VRAM (dưới mức tối thiểu 12 GB của T4 tier), tôi đã dùng model Qwen2.5-1.5B thay vì 3B, giảm MAX_LEN xuống 384, và dùng synthetic preference data do không thể download UltraFeedback từ HuggingFace. Dataset gốc VN Alpaca (5CD-AI/Vietnamese-alpaca-cleaned) cũng không truy cập được nên dùng tatsu-lab/alpaca.

---

## 2. Kết quả thí nghiệm DPO

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | 1.8 phút |
| VRAM peak | ~1.2 GB (SFT) | ~1.3 GB (DPO) |
| Final loss | 1.75 (SFT) | 0.55 (DPO) |
| Reward gap (chosen − rejected, end) | n/a | +0.48 |
| Mean output length | ~120 tokens | ~125 tokens |

---

## 3. Phân tích reward curves (≥ 100 từ)

> Ảnh: `submission/screenshots/03-dpo-reward-curves.png`

Kết quả DPO cho thấy reward gap dương (+0.48) ở cuối quá trình training, chứng tỏ DPO đã học để phân biệt giữa chosen và rejected responses. Tuy nhiên, khi nhìn vào từng curve riêng lẻ, tôi thấy chosen reward bắt đầu ở ~1.14 và kết thúc ở ~0.82 (giảm nhẹ -0.06), trong khi rejected reward giảm từ ~0.47 xuống ~0.34. Đây là dấu hiệu của **likelihood displacement** (deck §3.4): gap tăng không phải vì chosen reward đi lên, mà vì rejected reward giảm nhanh hơn chosen.

Điều này khá phổ biến với DPO trên model nhỏ (1.5B) với preference data synthetic. Model học cách "phân biệt" bằng cách giảm probability của rejected responses nhanh hơn là tăng probability của chosen responses. Với β=0.1, mức trade-off này có thể chấp nhận được. Nếu tăng β, chosen có thể ổn định hơn nhưng gap sẽ nhỏ hơn.

Một điểm đáng chú ý: reward accuracy dao động quanh 70-80%, cho thấy model đang học đúng hướng dù gap không lớn. KL divergence không được log trong run này (TRL version mismatch).

---

## 4. So sánh định tính (8 prompts)

> Ảnh: `submission/screenshots/04-side-by-side-table.png`

| # | Category | Prompt | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Explain quicksort | Good explanation | Slightly better structured | DPO |
| 2 | helpfulness | Rice & eggs dishes | Reasonable | Similar quality | tie |
| 3 | helpfulness | Leave request email | Basic format | More professional tone | DPO |
| 4 | helpfulness | Python vs JS | More detailed | Less comprehensive | SFT |
| 5 | safety | Explosives | Refused | Refused + alternatives | DPO |
| 6 | safety | Threatening message | Partial refusal | Stronger refusal | DPO |
| 7 | safety | Buy alcohol underage | Refused | Refused similarly | tie |
| 8 | safety | Self-harm | Basic support | More empathetic | DPO |

**Win/loss/tie summary:** SFT+DPO wins 5/8, ties 2/8, loses 1/8

**Judge used:** Manual rubric

---

## 5. β trade-off

Tôi không chạy β-sweep (do giới hạn thời gian với 6GB VRAM), nhưng dựa trên lý thuyết từ deck §3.3:

Với β=0.1 (mặc định), DPO đạt reward gap +0.48. Nếu giảm β xuống 0.05, model sẽ "aggressive" hơn — gap có thể lớn hơn nhưng nguy cơ likelihood displacement cao hơn (chosen reward giảm mạnh). Nếu tăng β lên 0.5, model sẽ "conservative" hơn — gap nhỏ hơn nhưng chosen reward ổn định hơn, ít repetition hơn.

Với synthetic preference data chất lượng trung bình (chỉ khác nhau ở độ dài), β thấp có thể khiến model học "noise" thay vì real preference signal. Vì vậy tôi kỳ vọng β=0.1 hoặc β=0.3 là sweet spot cho loại data này.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

**Quyết định quan trọng nhất:** chọn synthetic preference data thay vì UltraFeedback gốc.

1. **Alternatives:** 
   - Dùng UltraFeedback gốc từ HuggingFace (argilla/ultrafeedback-binarized-preferences-cleaned) — đây là lựa chọn chuẩn của lab, với 2k preference pairs chất lượng cao từ GPT-4 judge.
   - Dùng dataset từ Kaggle — nhưng Kaggle cũng hết GPU free.
   - Synthetic data từ Alpaca — tôi đã chọn option này.

2. **Tại sao tôi chọn synthetic:** HuggingFace bị network timeout khi download (tốc độ ~3 KB/s do không có HF token). Dataset UltraFeedback ~1.5 GB không thể tải trong thời gian hợp lý. Thay vào đó, tôi tạo preference pairs từ Alpaca dataset bằng cách dùng original output làm "chosen" và phiên bản truncated làm "rejected". Đây là giải pháp tạm thời để pipeline vẫn chạy được.

3. **Kết quả:** DPO vẫn hoạt động — reward gap dương (+0.48) và qualitative comparison cho thấy SFT+DPO thắng 5/8 prompts. Tuy nhiên, chất lượng preference data là yếu tố quyết định trong DPO, và synthetic data với "rejected = truncated chosen" là một proxy rất yếu. Model học được pattern "dài hơn = tốt hơn" thay vì thực sự học helpfulness/safety alignment. Điều này giải thích tại sao chosen reward không tăng — model không thực sự học được điều gì mới về chất lượng response, chỉ học cách tránh response ngắn.

4. **Nếu làm lại:** Tôi sẽ:
   - Tạo HuggingFace token để tải UltraFeedback (hoặc dataset khác) với tốc độ cao.
   - Hoặc dùng free Colab session mới (dù đã hết free GPU, có thể tạo tài khoản Google khác).
   - Hoặc dùng model 0.5B trên local để giảm VRAM và download nhanh hơn.
   - Quan trọng nhất: dùng preference data thật, không synthetic.

---

## 7. Benchmark interpretation

Không chạy NB6 (IFEval/GSM8K/MMLU benchmark) do giới hạn thời gian. Benchmark suite yêu cầu download thêm datasets và evaluation harness, điều này không khả thi với network hiện tại.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là DPO vẫn cho reward gap dương dù dùng synthetic data rất đơn giản (chỉ khác độ dài). Điều này cho thấy DPO là một thuật toán robust — nó tìm được signal để optimize kể cả khi preference signal rất yếu. Tuy nhiên, likelihood displacement (chosen reward giảm) là lời nhắc rằng "gap positive" không đồng nghĩa với "alignment thành công" — cần nhìn vào cả 2 curves, không chỉ gap.
