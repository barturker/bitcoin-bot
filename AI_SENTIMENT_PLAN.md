# AI Sentiment Katmanı - Yapılacaklar

## Genel Bakış
Multi-source sentiment analizi sistemi. Birden fazla LLM ve veri kaynağı kullanarak piyasa duygusunu analiz edip trading modeline input olarak verecek.

---

## API Keys (Toplanacak)

- [ ] **Groq** - https://console.groq.com
- [ ] **Google Gemini** - https://aistudio.google.com
- [ ] **HuggingFace** - https://huggingface.co/settings/tokens
- [ ] **CryptoPanic** - https://cryptopanic.com/developers/api
- [ ] **Binance** - https://www.binance.com/en/my/settings/api-management

`.env` dosyasına eklenecek:
```
BINANCE_API_KEY=
BINANCE_API_SECRET=
GROQ_API_KEY=
GOOGLE_AI_API_KEY=
HUGGINGFACE_API_KEY=
CRYPTOPANIC_API_KEY=
```

---

## Veri Kaynakları

### 1. CryptoPanic API (Haberler)
- Kripto haberleri agregator
- Ücretsiz tier: 1000 req/gün
- Bullish/bearish etiketli haberler

### 2. Fear & Greed Index
- https://alternative.me/crypto/fear-and-greed-index/
- Ücretsiz, API mevcut
- 0-100 arası skor

### 3. Twitter/X Sentiment
- Kripto hashtag'leri analizi
- Ücretsiz API kısıtlı, alternatif: nitter scraping

### 4. Reddit
- r/bitcoin, r/cryptocurrency
- PRAW kütüphanesi ile

---

## LLM Modelleri

### 1. Groq (Ana Model)
- Model: Llama 3.1 70B
- Limit: 30 req/dk (yeterli)
- Kullanım: Haber analizi, genel sentiment

### 2. Google Gemini
- Model: Gemini Pro
- Limit: 60 req/dk
- Kullanım: İkinci görüş, doğrulama

### 3. HuggingFace FinBERT
- Model: ProsusAI/finbert
- Limit: Yüksek
- Kullanım: Finansal metin için özel eğitilmiş

---

## Sistem Mimarisi

```
┌─────────────────── VERİ KAYNAKLARI ───────────────────┐
│                                                        │
│  CryptoPanic API ──► Haberler (son 1 saat)            │
│  Fear & Greed    ──► Index değeri                     │
│  Reddit PRAW     ──► Top posts sentiment              │
│                                                        │
└────────────────────────┬───────────────────────────────┘
                         ▼
┌─────────────────── LLM ANALİZ ────────────────────────┐
│                                                        │
│  Groq (Llama 3.1)    ──► sentiment_1 (-1 to +1)       │
│  Google Gemini       ──► sentiment_2 (-1 to +1)       │
│  FinBERT             ──► sentiment_3 (-1 to +1)       │
│                                                        │
└────────────────────────┬───────────────────────────────┘
                         ▼
┌─────────────────── BİRLEŞTİRME ───────────────────────┐
│                                                        │
│  weighted_sentiment = 0.4*s1 + 0.35*s2 + 0.25*s3      │
│  confidence = std_dev(sentiments) # düşük = hemfikir  │
│                                                        │
│  Çıktı: sentiment (-1 to +1), confidence (0 to 1)     │
│                                                        │
└────────────────────────┬───────────────────────────────┘
                         ▼
                [Trading Model Input]
```

---

## Dosya Yapısı

```
src/
├── sentiment/
│   ├── __init__.py
│   ├── data_sources.py      # CryptoPanic, Fear&Greed, Reddit
│   ├── llm_analyzers.py     # Groq, Gemini, FinBERT
│   ├── aggregator.py        # Sentimentleri birleştir
│   └── sentiment_engine.py  # Ana modül
├── paper_trading.py         # Paper trading sistemi
└── live_trading.py          # Canlı trading (ileride)
```

---

## Yapılacaklar Listesi

### Faz 1: Temel Altyapı
- [ ] `.env` dosyası ve config yapısı
- [ ] API bağlantı testleri
- [ ] Rate limit yönetimi

### Faz 2: Veri Kaynakları
- [ ] CryptoPanic entegrasyonu
- [ ] Fear & Greed Index entegrasyonu
- [ ] Reddit PRAW entegrasyonu

### Faz 3: LLM Entegrasyonu
- [ ] Groq client
- [ ] Gemini client
- [ ] FinBERT (HuggingFace)
- [ ] Prompt engineering (sentiment analizi için)

### Faz 4: Birleştirme
- [ ] Weighted average hesaplama
- [ ] Confidence score
- [ ] Fallback mekanizması (API çökerse)

### Faz 5: Trading Entegrasyonu
- [ ] Sentiment'i model input'una ekle
- [ ] Paper trading'de test et
- [ ] Dashboard'a sentiment göstergesi ekle

---

## Örnek Prompt (LLM için)

```
Analyze the following crypto news headlines and provide a sentiment score.

Headlines:
{headlines}

Respond with ONLY a JSON object:
{
  "sentiment": <float between -1 (very bearish) and 1 (very bullish)>,
  "confidence": <float between 0 and 1>,
  "reasoning": "<brief explanation>"
}
```

---

## Notlar

- Her saat başı çalışacak (model 1h timeframe)
- Rate limit'lere dikkat
- API çökerse son bilinen sentiment kullan
- Sentiment değişimi ani olursa (>0.5 fark) log'la

---

## Durum

**Şu anki durum:** Training devam ediyor
**Sonraki adım:** Training bitince paper trading + sentiment modülü yazılacak
