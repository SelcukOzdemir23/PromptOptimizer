# EvoPrompt Optimizer

> **Proje Kodu: HZ-2026-001** — Hesapsal Zeka Dersi Vize Ödevi  
> **Neuroevolution Tabanlı LLM Prompt Optimizasyonu ile Haber Sınıflandırma**

Büyük dil modellerinin parametrelerini dondurarak yalnızca modele gönderilen doğal dil talimatını (prompt) genetik algoritma ile optimize eden bir sistem. AG News veri seti üzerinde 4 sınıflı haber sınıflandırma görevinde en yüksek doğruluğu sağlayan prompt'u evrimsel süreçle otomatik keşfeder.

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Proje klasörüne gir
cd PromptOptimizer

# 2. Sanal ortamı aktifleştir
source venv/bin/activate

# 3. API bağlantısını test et
python tests/test_api_health.py

# 4. Tam pipeline'ı çalıştır
python src/main.py
```

**Gereksinimler:**
- Python 3.10+
- Google Gemini API key (`.env` dosyasında tanımlı olmalı)
- [Ücretsiz API key al](https://aistudio.google.com/app/apikey)

---

## 📋 Proje Nasıl Çalıştırılır

### Adım 1: API Key Ayarla

`.env` dosyasını aç ve `GEMINI_API_KEY` değerini kendi API anahtarın ile değiştir:

```
GEMINI_API_KEY=AIzaSy...senin_keyin_buraya
```

API key'in yoksa: [Google AI Studio](https://aistudio.google.com/app/apikey) adresinden ücretsiz al.

### Adım 2: Sanal Ortamı Aktif Et

```bash
source venv/bin/activate
```

### Adım 3: API Bağlantısını Test Et

```bash
python tests/test_api_health.py
```

Bu komut şunları kontrol eder:
- ✅ API key tanımlı mı?
- ✅ SDK doğru yüklenmiş mi?
- ✅ Model erişilebilir mi?
- ✅ Basit bir istek başarılı mı?

### Adım 4: Pipeline'ı Çalıştır

```bash
python src/main.py
```

**Pipeline şu adımları otomatik yapar:**
1. AG News veri setini yükler ve eğitim/test olarak böler
2. İnsan yazımı başlangıç prompt'unun doğruluğunu ölçer
3. Genetik algoritma ile prompt'u optimize eder (her jenerasyonda popülasyondaki her birey için mini-batch değerlendirme)
4. En iyi evolved prompt'u test setinde değerlendirir
5. Grafikler, raporlar ve CSV istatistik dosyalarını `outputs/` klasörüne kaydeder

### Adım 5: Çıktıları İncele

Çalışma tamamlandığında `outputs/` klasöründe:

| Dosya | Açıklama |
|-------|----------|
| `accuracy_curve.png` | Jenerasyonlar boyunca doğruluk değişim grafiği |
| `best_prompt.txt` | En iyi evolved prompt (okunabilir format) |
| `best_prompt.json` | En iyi prompt + metadata (JSON) |
| `report.txt` | Tam evrim raporu (akademik kullanım için) |
| `generation_stats.csv` | Her jenerasyonun detaylı istatistikleri |
| `experiment_summary.csv` | Deney özeti — tüm metrikler tek dosyada |
| `population_analysis.csv` | Son popülasyondaki tüm prompt'lar ve skorları |

---

## ⚙️ Yapılandırma Değişkenleri (.env)

Tüm ayarlar `.env` dosyasından okunur. Her değişkenin anlamı ve yüksek/düşük değerlerin etkisi:

### 🔑 API & Model

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `GEMINI_API_KEY` | _(gerekli)_ | Google Gemini API anahtarınız. |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Kullanılacak model. |

#### GEMINI_MODEL — Model Seçimi

| Model | Hız | Maliyet | Ne İçin |
|-------|-----|---------|---------|
| `gemini-2.5-flash-lite` | ⚡ En hızlı | 💰 En ucuz | **Önerilen — bu proje için ideal** |
| `gemini-2.5-flash` | Hızlı | Düşük | Daha yüksek doğruluk, daha yavaş |
| `gemini-2.5-pro` | Yavaş | Daha yüksek | En iyi akıl yürütme kalitesi |

---

### 🧬 Evrim Parametreleri

#### POPULATION_SIZE

```
POPULATION_SIZE=20
```

Her jenerasyondaki aday prompt sayısı.

| Değer | Etki |
|-------|------|
| **Düşük (5-10)** | Her jenerasyon daha hızlı, ancak erken yakınsama riski. Optimal prompt'u kaçırabilir. |
| **Varsayılan (20)** | İyi denge. Farklı ifade varyasyonlarını keşfeder, aşırı API çağrısı yapmaz. |
| **Yüksek (40-100)** | Daha iyi keşif, optimal prompt bulma şansı artar. **Çok yavaş ve pahalı** — her birey 50 API çağrısı gerektirir. |

**Trade-off:** `POPULATION_SIZE × JENERASYON × MINI_BATCH_SIZE` = toplam API çağrısı. 20 × 10 × 50 = **~10.000 API çağrısı** minimum.

---

#### GENERATIONS

```
GENERATIONS=10
```

Kaç jenerasyon evrim çalıştırılacağı.

| Değer | Etki |
|-------|------|
| **Düşük (3-5)** | Hızlı çalışma, ancak prompt henüz en iyi haline ulaşmamış olabilir. İlk iyileşmeleri görürsün ama ince ayar eksik kalır. |
| **Varsayılan (10)** | Prompt'un stabilize olması için yeterli. Doğruluk genelde 6-8. jenerasyonda plato çizer. |
| **Yüksek (20-50)** | 15 jenerasyondan sonra azalan verim. Yakınsama davranışını izlemek için faydalı. Maliyeti ciddi şekilde artırır. |

**İpucu:** `accuracy_curve.png` grafiğine bak. Eğri son jenerasyonda hâlâ yükseliyorsa bu değeri artır.

---

#### MUTATION_PROBABILITY

```
MUTATION_PROBABILITY=0.2
```

Bir prompt'un mutasyona uğrama olasılığı (0.0-1.0). Mutasyon, bir kelimeyi WordNet eş anlamlısı ile değiştirir.

| Değer | Etki |
|-------|------|
| **Düşük (0.05-0.1)** | Muhafazakar. Prompt'lar yavaş değişir. Başlangıç prompt'u zaten güçlüyse iyi. Risk: yavaş keşif, yerel optimumda sıkışma. |
| **Varsayılan (0.2)** | Dengeli. Varyasyon keşfi yeterli ama iyi prompt'ları bozmaz. |
| **Yüksek (0.4-0.8)** | Agresif. Çok kelime değişir, çok farklı prompt'lar oluşur. Erken keşif için iyi ama iyi prompt'ları bozabilir. |

**Analoji:** Mutasyon "eş anlamlı deneme" gibidir. Çok az = hiç yeni kelime denemezsin. Çok fazla = iyi bir cümleyi oturtramazsın.

---

#### CROSSOVER_PROBABILITY

```
CROSSOVER_PROBABILITY=0.8
```

İki ebeveyn prompt'unun çaprazlanma olasılığı (0.0-1.0).

| Değer | Etki |
|-------|------|
| **Düşük (0.1-0.3)** | Nadir rekombinasyon. Evrim çoğunlukla mutasyona güvenir. İyi fikirleri birleştirmek yavaş olur. |
| **Varsayılan (0.8)** | **Önerilen.** Yüksek çaprazlama, iki iyi prompt'un en iyi kısımlarını birleştirmeye izin verir. |
| **Yüksek (0.9-1.0)** | Neredeyse her zaman çaprazlar. Etkili olabilir ama mükemmel bir prompt yapısını bozabilir. |

**Kural:** `CROSSOVER_PROBABILITY` genellikle `MUTATION_PROBABILITY`'den yüksek olmalı (0.8 vs 0.2 klasik oran).

---

#### TOURNAMENT_SIZE

```
TOURNAMENT_SIZE=3
```

Her turnuva seçiminde yarışan birey sayısı.

| Değer | Etki |
|-------|------|
| **Düşük (2)** | Zayıf seçim baskısı. Ortanca prompt'lar bile üreyebilir. Çeşitlilik korunur ama yakınsama yavaş olur. |
| **Varsayılan (3)** | İyi denge. En iyi 1/3'ün makul şansı var, yine de kaliteli prompt'lar katılabiliyor. |
| **Yüksek (5-10)** | Güçlü seçim baskısı. Sadece en iyiler ürer. Hızlı yakınsama ama çeşitlilik kaybı riski. |

---

#### MINI_BATCH_SIZE

```
MINI_BATCH_SIZE=50
```

Her bireyin fitness (doğruluk) değerlendirmesi için kullanılan haber başlığı sayısı.

| Değer | Etki |
|-------|------|
| **Düşük (10-20)** | Çok hızlı. Ama doğruluk tahmini gürültülü — bir prompt bir batch'te %80, diğerinde %40 alabilir. Güvenilmez seçim. |
| **Varsayılan (50)** | Makul doğruluk tahmini (~%7 hata payı). Hız/kalite dengesi iyi. |
| **Yüksek (100-500)** | Daha doğru fitness değerlendirmesi. Daha iyi seçim kararları. **Ama** her birey daha fazla API çağrısı gerektirir, evrim çok yavaşlar. |

**API maliyetine etkisi:** **En önemli parametre.** `MINI_BATCH_SIZE=50` ile her birey 50 API çağrısı. `MINI_BATCH_SIZE=200` ile 4× daha fazla.

**Formül:** Toplam API çağrısı ≈ `POPULATION_SIZE × JENERASYON × MINI_BATCH_SIZE`  
Varsayılanlarla: 20 × 10 × 50 = **~10.000 API çağrısı**

---

### ⏱️ Rate Limiting

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `API_CALL_DELAY` | `4.0` | Her API çağrısı arası bekleme süresi (saniye). |
| `API_MAX_RETRIES` | `3` | Geçici hatalarda yeniden deneme sayısı. |
| `API_BACKOFF_FACTOR` | `2.0` | Üstel geri çekilme çarpanı. |

#### Rate Limit Ayarlama

| Durum | Değişiklik |
|-------|-----------|
| **429 hatası alıyorsan** | `API_CALL_DELAY`'i 6-8 saniyeye çıkar |
| **Çok yavaş, hata yok** | `API_CALL_DELAY`'i 2-3 saniyeye düşür (kotan yetiyorsa) |
| **Sık geçici hatalar** | `API_MAX_RETRIES`'ı 5'e çıkar |

**Uyarı:** Gemini API free tier'ında sıkı rate limitleri var (Flash-Lite için 15 RPM). `API_CALL_DELAY=4.0` ile ~15 çağrı/dakika, sınıra yakın. 4.0'ın altına düşürme.

---

### 📊 Veri Bölme

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `TEST_SPLIT_RATIO` | `0.5` | Test veri setinin ne kadarı final test olarak ayrılır. |
| `RANDOM_SEED` | `42` | Tekrarlanabilir sonuçlar için rastgelelik tohumu. |
| `TEST_SAMPLE_SIZE` | `50` | Final test değerlendirmesinde kullanılacak örnek sayısı. `-1` = tamamı (çok pahalı!). |

---

## 📁 Proje Yapısı

```
PromptOptimizer/
├── src/
│   ├── config.py            # Merkezi yapılandırma (.env okur)
│   ├── data_loader.py       # AG News yükleme, bölme, önbellekleme
│   ├── llm_interface.py     # Gemini API wrapper (yeni google-genai SDK)
│   ├── evolution.py         # DEAP genetik algoritma motoru
│   ├── visualizer.py        # Grafikler, raporlar, CSV istatistikler
│   └── main.py              # Tam pipeline orkestratörü
├── tests/
│   ├── test_api_health.py   # API bağlantı ve sağlık kontrolü
│   └── test_e2e.py          # 23 birim testi
├── data/
│   ├── raw/                 # (ham veri için ayrılmış)
│   └── processed/           # Önbelleklenmiş dev_pool.csv ve test_set.csv
├── dataset/                 # Kaynak AG News CSV dosyaları
├── outputs/                 # Çalışma sonrası oluşan sonuçlar
│   ├── accuracy_curve.png   # Doğruluk değişim grafiği
│   ├── best_prompt.txt      # En iyi evolved prompt (okunabilir)
│   ├── best_prompt.json     # En iyi prompt + metadata
│   ├── report.txt           # Tam evrim raporu
│   ├── generation_stats.csv # Her jenerasyonun istatistikleri
│   ├── experiment_summary.csv # Deney özeti - tüm metrikler
│   └── population_analysis.csv # Son popülasyon analizi
├── .env                     # Yapılandırman (ASLA commit'leme!)
├── .env.example             # .env şablonu
├── .gitignore
├── requirements.txt
├── venv/                    # Python sanal ortamı
└── README.md
```

---

## 🧪 Testler

```bash
# Sanal ortamı aktifleştir
source venv/bin/activate

# Tüm birim testlerini çalıştır (23 test)
pytest tests/test_e2e.py -v

# API sağlık kontrolü
python tests/test_api_health.py
```

---

## � Çıktı Dosyaları Detayları

### CSV İstatistik Dosyaları

**`generation_stats.csv`** — Her jenerasyonun detaylı istatistikleri:
```
generation, avg_fitness, max_fitness, min_fitness, avg_fitness_pct, max_fitness_pct, min_fitness_pct, fitness_spread
0, 0.450000, 0.520000, 0.380000, 45.00%, 52.00%, 38.00%, 0.140000
1, 0.510000, 0.600000, 0.420000, 51.00%, 60.00%, 42.00%, 0.180000
```

**`experiment_summary.csv`** — Tüm deney metrikleri tek dosyada:
- Model, popülasyon, jenerasyon, mutasyon/çaprazlama olasılıkları
- Başlangıç, en iyi dev ve test doğrulukları
- İyileşme yüzdesi, yakınsama jenerasyonu
- Toplam API çağrısı tahmini

**`population_analysis.csv`** — Son popülasyondaki her birey:
- Sıralama, fitness skoru, prompt uzunluğu
- Her prompt'un tam metni

---

## 📚 Akademik Kaynaklar

Bu proje aşağıdaki çalışmalardan ilham almıştır:

- **Sprig (Nisan 2026)** — Düzenleme operasyonları tabanlı genetik algoritma ile sistem prompt optimizasyonu
- **GEPA (2025-2026)** — Genetik Evrimsel Prompt Optimizasyonu, GA'ların RL yöntemleriyle rekabet edebildiğini gösterir
- **RoboPhd (Nisan 2026)** — LLM destekli evrimsel ajan optimizasyonu

---

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji |
|---------|-----------|
| Dil | Python 3.10+ |
| LLM API | Google Gemini (yeni `google-genai` SDK ≥ 1.0.0) |
| Model | Gemini 2.5 Flash-Lite |
| Genetik Algoritma | DEAP 1.4+ |
| Eş Anlamlı Arama | NLTK WordNet |
| Veri İşleme | Pandas, scikit-learn |
| Görselleştirme | Matplotlib |
| Test | pytest |

---

## ⚠️ Önemli Notlar

- **API Key Kotası:** Free tier'ın günlük/aylık limitleri vardır. 429 hatası alırsan bekle veya billing'i aç.
- **Maliyet:** Varsayılan ayarlarla ~10.000 API çağrısı yapılır. Free tier'da bu sınıra yakındır.
- **Tekrarlanabilirlik:** `RANDOM_SEED` ayarla ki her çalışmada aynı veri bölünsün. Evrimde rastgelelik var, sonuçlar değişebilir.
- **`.env` dosyası:** Asla `.env` dosyasını commit'leme. `.env.example`'ı şablon olarak kullan.
