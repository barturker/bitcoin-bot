# AI Sentiment Katmanı - Yapılacaklar

## Genel Bakış
Multi-source sentiment analizi sistemi. Birden fazla LLM ve veri kaynağı kullanarak piyasa duygusunu analiz edip trading modeline input olarak verecek.

---

## API Keys (Toplanacak)

### Zorunlu (Ücretsiz)
- [ ] **Binance** - https://www.binance.com/en/my/settings/api-management
- [ ] **Groq** - https://console.groq.com
- [ ] **Google Gemini** - https://aistudio.google.com
- [ ] **HuggingFace** - https://huggingface.co/settings/tokens
- [ ] **CryptoPanic** - https://cryptopanic.com/developers/api

### Whale & On-Chain (Ücretsiz)
- [ ] **Whale Alert** - https://whale-alert.io/api (ücretsiz tier)
- [ ] **Glassnode** - https://studio.glassnode.com/ (bazı metrikler ücretsiz)
- [ ] **Blockchain.com** - API key gerekmiyor

### Sosyal Medya (Opsiyonel)
- [ ] **Reddit** - https://www.reddit.com/prefs/apps (PRAW için)
- [ ] **Twitter/X** - Paralı, alternatif: Nitter scraping

`.env` dosyasına eklenecek:
```
# Binance
BINANCE_API_KEY=
BINANCE_API_SECRET=

# LLM APIs
GROQ_API_KEY=
GOOGLE_AI_API_KEY=
HUGGINGFACE_API_KEY=

# Data Sources
CRYPTOPANIC_API_KEY=
WHALE_ALERT_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=

# Alerts (Opsiyonel)
TELEGRAM_BOT_TOKEN=
DISCORD_WEBHOOK_URL=
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

### 5. Whale Alert (YENİ)
- https://whale-alert.io/
- Büyük BTC transferlerini takip
- Exchange'e giriş = satış sinyali
- Exchange'den çıkış = HODL sinyali
- Ücretsiz API mevcut

### 6. On-Chain Data (YENİ)
- Glassnode (bazı metrikler ücretsiz)
- Exchange inflow/outflow
- Active addresses
- MVRV ratio

### 7. Ünlü İsimler Twitter Takibi (YENİ)
Takip edilecek hesaplar:
- @elonmusk - Elon Musk (büyük etki)
- @saborlorsaylor - Michael Saylor (Bitcoin maximalist)
- @VitalikButerin - Vitalik (Ethereum ama crypto genel)
- @caborlosz - CZ Binance
- @brian_armstrong - Coinbase CEO
- @APompliano - Anthony Pompliano
- Scraping: Nitter veya Twitter API

### 8. Whale Wallet Tracking (YENİ)
Takip edilecek cüzdanlar:
- Satoshi cüzdanları (hareket ederse büyük haber)
- MicroStrategy cüzdanı
- Tesla cüzdanı
- Büyük exchange cold wallet'ları
- Top 100 BTC holder adresleri
- Kaynak: Blockchain.com, Bitinfocharts

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
┌──────────────────────── VERİ KAYNAKLARI ────────────────────────┐
│                                                                  │
│  ┌─── HABERLER ───┐  ┌─── SOSYAL ───┐  ┌─── ON-CHAIN ───┐      │
│  │ CryptoPanic    │  │ Twitter      │  │ Whale Alert    │      │
│  │ Fear & Greed   │  │ Reddit       │  │ Exchange Flow  │      │
│  │ Google News    │  │ Ünlü İsimler │  │ Whale Wallets  │      │
│  └────────────────┘  └──────────────┘  └────────────────┘      │
│                                                                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────── LLM ANALİZ ─────────────────────────────┐
│                                                                  │
│  Groq (Llama 3.1)    ──► news_sentiment                         │
│  Google Gemini       ──► social_sentiment                       │
│  FinBERT             ──► financial_sentiment                    │
│                                                                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────── WHALE SİNYALLERİ ───────────────────────┐
│                                                                  │
│  Exchange'e büyük giriş (>1000 BTC)  ──► SATIŞ sinyali 🔴       │
│  Exchange'den büyük çıkış            ──► HODL sinyali 🟢        │
│  Ünlü cüzdan hareketi                ──► ALERT! ⚠️              │
│  Elon tweet                          ──► Anlık analiz 🐦        │
│                                                                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────── BİRLEŞTİRME ────────────────────────────┐
│                                                                  │
│  final_sentiment = weighted_average(all_sources)                │
│  whale_signal = analyze_whale_activity()                        │
│  vip_signal = check_vip_tweets()                                │
│  confidence = model_agreement_score()                           │
│                                                                  │
│  Çıktı:                                                         │
│  - sentiment: -1 (bearish) to +1 (bullish)                      │
│  - whale_signal: -1 (selling) to +1 (accumulating)              │
│  - vip_alert: bool (ünlü biri tweet attı mı?)                   │
│  - confidence: 0 to 1                                           │
│                                                                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
                     [Trading Model Input]
```

---

## Dosya Yapısı

```
src/
├── sentiment/
│   ├── __init__.py
│   ├── news_sources.py      # CryptoPanic, Fear&Greed, Google News
│   ├── social_sources.py    # Twitter, Reddit, Ünlü isimler
│   ├── whale_tracker.py     # Whale Alert, Exchange flow, Wallet tracking
│   ├── llm_analyzers.py     # Groq, Gemini, FinBERT
│   ├── aggregator.py        # Tüm sinyalleri birleştir
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

### Faz 2: Haber Kaynakları
- [ ] CryptoPanic entegrasyonu
- [ ] Fear & Greed Index entegrasyonu
- [ ] Google News crypto haberleri

### Faz 3: Sosyal Medya
- [ ] Reddit PRAW entegrasyonu
- [ ] Twitter/Nitter scraping
- [ ] Ünlü isim listesi ve takip sistemi

### Faz 4: Whale Tracking (YENİ)
- [ ] Whale Alert API entegrasyonu
- [ ] Exchange inflow/outflow takibi
- [ ] Ünlü cüzdan adresleri listesi
- [ ] Büyük transfer alert sistemi

### Faz 5: LLM Entegrasyonu
- [ ] Groq client
- [ ] Gemini client
- [ ] FinBERT (HuggingFace)
- [ ] Prompt engineering (sentiment analizi için)

### Faz 6: Birleştirme
- [ ] Weighted average hesaplama
- [ ] Whale signal scoring
- [ ] VIP tweet alert sistemi
- [ ] Confidence score
- [ ] Fallback mekanizması (API çökerse)

### Faz 7: Trading Entegrasyonu
- [ ] Tüm sinyalleri model input'una ekle
- [ ] Paper trading'de test et
- [ ] Dashboard'a sentiment + whale göstergesi ekle
- [ ] Real-time alert sistemi (Telegram/Discord)

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
