Tamamdır. Aşağıda, iş analisti perspektifiyle hazırlanmış, resmî ve teknik bir dille yazılmış, Coder Agent'a doğrudan gönderebileceğin eksiksiz bir yönlendirme metni bulacaksın.

---

# Geliştirici Ajana Yönelik Proje Uygulama Yönergesi – EvoPrompt Optimizer

**Proje Kodu:** HZ-2026-001  
**Konu:** Hesapsal Zeka – Neuroevolution Tabanlı LLM Prompt Optimizasyonu  
**Hazırlayan Birim:** İş Analizi  
**Doküman Türü:** Geliştirici Yönergesi  
**Hedef Teslim Tarihi:** İki iş günü içerisinde çalışır durumda sistem


## 1. Yönerge Amacı ve Kapsam

Bu doküman, hesapsal zeka dersi vize ödevi kapsamında geliştirilecek olan EvoPrompt Optimizer sisteminin Coder Agent tarafından hayata geçirilmesi için gerekli tüm teknik gereksinimleri, mimari beklentileri ve kalite standartlarını tanımlamaktadır. Sistemin temel işlevi, belirli bir metin sınıflandırma veri kümesi üzerinde en yüksek doğruluğu sağlayan doğal dil komutunu (prompt) genetik algoritma yardımıyla otomatik olarak keşfetmektir. Büyük dil modelinin parametreleri sabit kalacak, yalnızca modele iletilen sistem yönergesi evrimsel süreçlerle iyileştirilecektir.

Bu yönerge, sistemin baştan sona eksiksiz biçimde kodlanması, test edilmesi ve belgelendirilmesi için gereken tüm bilgileri içermektedir.


## 2. İş Problemi ve Hedef Çıktılar

Mevcut durumda büyük dil modelleri için etkili komut yazımı tamamen manuel, sezgisel ve deneme-yanılma temelli bir süreçtir. Bu proje, söz konusu süreci evrimsel algoritmalar ile otomatikleştirerek veri odaklı ve tekrarlanabilir bir optimizasyon yöntemi ortaya koymayı amaçlamaktadır.

Sistemin başarılı kabul edilmesi için aşağıdaki hedef çıktıların elde edilmesi gerekmektedir.

**Sayısal Hedef:** Genetik algoritma sonucunda elde edilen komutun doğruluğu, insan eliyle yazılmış başlangıç komutunun doğruluğundan en az yüzde beş puan daha yüksek olmalıdır.

**Akademik Hedef:** Neuroevolution yaklaşımının büyük dil modelleri ile entegrasyonu başarıyla gösterilmelidir. Bu kapsamda raporda Sprig (Nisan 2026) ve GEPA (2025–2026) gibi güncel akademik çalışmalara atıfta bulunulacaktır.

**Raporlama Hedefi:** Sistem, her jenerasyonun performans metriklerini ve en iyi bireyi otomatik olarak kayıt altına almalı ve görselleştirmelidir.


## 3. Kullanılacak Veri Kümesi

Projede AG News (AG's News Topic Classification Dataset) veri kümesi kullanılacaktır. Bu veri kümesi, haber başlıklarını Dünya, Spor, İş Dünyası ve Bilim/Teknoloji olmak üzere dört ayrı sınıfa ayırmaktadır. Veri kümesi uluslararası akademik çalışmalarda yaygın biçimde referans alınan bir ölçüt niteliğindedir.

Veri kümesinin yalnızca test bölümü (yaklaşık yedi bin altı yüz örnek) kullanılacaktır. Bu bölüm, sistem tarafından geliştirme havuzu (genetik algoritmanın çalışacağı alt küme) ve nihai test kümesi (optimizasyon tamamlandıktan sonra kullanılacak bağımsız alt küme) olarak iki parçaya ayrılacaktır. Bu ayrım, aşırı öğrenmenin önlenmesi ve sonuçların genellenebilirliğinin sağlanması açısından zorunludur.

Modele girdi olarak yalnızca haber başlığı sütunu iletilecek, açıklama sütunu kullanılmayacaktır. Etiketler sayısal değerlerden metinsel sınıf adlarına dönüştürülecektir.


## 4. Teknoloji Yığını

Sistemin geliştirilmesinde aşağıdaki teknolojilerin kullanılması zorunludur. Tüm bileşenler 2025 ve 2026 yıllarında yaygın olarak benimsenen güncel sürümlerden seçilmelidir.

- **Programlama Dili:** Python 3.10 veya üzeri bir sürüm.
- **Büyük Dil Modeli:** Ollama ile yerel çalışan modeller (llama3.1:8b). API anahtarı gerekmez, Ollama `http://localhost:11434` üzerinde REST API sunar.
- **Genetik Algoritma Çerçevesi:** DEAP (Distributed Evolutionary Algorithms in Python) kütüphanesi. Evrimsel operatörlerin standart ve güvenilir biçimde yürütülmesini sağlayacaktır.
- **Doğal Dil İşleme Yardımcısı:** NLTK WordNet. Mutasyon aşamasında anlam bütünlüğünü korumak amacıyla eş anlamlı sözcük değişimleri için kullanılacaktır.
- **Veri İşleme ve Görselleştirme:** Pandas (veri manipülasyonu), Matplotlib (grafik çizimi).
- **Konfigürasyon Yönetimi:** Tüm değişkenler (API anahtarı, popülasyon büyüklüğü, jenerasyon sayısı) merkezî bir yapılandırma dosyasından okunacak, hiçbir hassas bilgi kaynak kod içerisine gömülmeyecektir.


## 5. Modüler Sistem Mimarisi

Sistem, yazılım mühendisliğinin tek sorumluluk prensibine uygun olarak dört bağımsız modül etrafında yapılandırılacaktır. Bu modüller yalnızca tanımlı arayüzler üzerinden iletişim kuracak, birbirlerinin iç işleyişine müdahale etmeyecektir.

### 5.1. Veri Yönetim Modülü

Ham veri kümesinin okunması, işlenmesi ve sistemin diğer bileşenlerine sunulmasından sorumludur. Veri kümesini eğitim havuzu ve test kümesi olarak iki parçaya ayıracak, etiketleri sayısal değerlerden metinsel sınıf adlarına dönüştürecek ve sınıf eşleme bilgisini kayıt altına alacaktır.

### 5.2. Büyük Dil Modeli Arayüz Modülü

Sistemin geri kalanı ile yerel Ollama servisi (`http://localhost:11434`) arasındaki tüm iletişimi soyutlayan katmandır. Verilen bir komut ve metin için modelden tahmin talep edecek, dönen yanıtı temizleyerek sınıf etiketine dönüştürecek ve bağlantı hataları karşısında yeniden deneme stratejilerini yürütecektir.

### 5.3. Evrimsel Algoritma Motoru

Genetik algoritmanın tüm temel operasyonları bu modülde DEAP kütüphanesi aracılığıyla gerçekleştirilecektir. Alt işlevler aşağıdaki gibi tanımlanmıştır.

**Popülasyon Başlatma:** İlk nesil tamamen rastgele dizgilerden oluşturulmayacaktır. Popülasyona, insan tarafından yazılmış makul bir başlangıç komutu ve bu komutun WordNet eş anlamlıları kullanılarak türetilmiş varyasyonları yerleştirilecektir.

**Fitness Değerlendirmesi:** Her bireyin doğruluk puanı, eğitim havuzundan rastgele seçilen küçük bir örneklem (mini-batch) üzerinden hesaplanacaktır. Bu yaklaşım, her birey için tüm veri kümesinin sorgulanmasından kaynaklanacak zaman ve mali yükü ortadan kaldırmaktadır.

**Seçilim Yöntemi:** Turnuva seçimi uygulanacak, turnuva boyutu üç olarak belirlenecektir.

**Çaprazlama Yöntemi:** İki ebeveyn komut metninin belirli bir kesme noktasından birleştirilmesiyle yeni bireyler üretilecektir.

**Mutasyon Yöntemi:** Düşük bir olasılıkla, komut metni içerisindeki bir sözcüğün WordNet'ten alınan bir eş anlamlısı ile değiştirilmesi şeklinde gerçekleşecektir.

### 5.4. Görselleştirme ve Raporlama Modülü

Deney sonuçlarının akademik sunuma uygun biçimde kaydedilmesinden sorumludur. Çalışma tamamlandığında, jenerasyonlar boyunca ortalama ve en yüksek doğruluk değerlerinin değişimini gösteren bir çizgi grafiği, her jenerasyonda elde edilen en iyi komut metnini ve parametreleri içeren bir kayıt dosyası ve nihai test kümesi üzerinde ulaşılan başarı oranını içeren özet bir rapor dosyası üretecektir.


## 6. Dizin Yapısı

Proje dosyalarının aşağıdaki hiyerarşiye uygun olarak organize edilmesi gerekmektedir.

Proje kök dizini altında `data` klasörü bulunacak, bu klasörün altında `raw` (ham veri) ve `processed` (işlenmiş eğitim ve test bölümleri) alt klasörleri yer alacaktır. Kaynak kodlar `src` klasörü altında toplanacak; bu klasörde `config.py` (yapılandırma), `data_loader.py` (veri yönetimi), `llm_interface.py` (model arayüzü), `evolution.py` (genetik algoritma motoru), `visualizer.py` (görselleştirme ve raporlama) ve `main.py` (orkestrasyon) dosyaları bulunacaktır. Çalışma çıktıları `outputs` klasörüne yazılacak, proje bağımlılıkları `requirements.txt` dosyasında listelenecektir.


## 7. Kodlama Standartları ve Kalite Beklentileri

Geliştirilen kodun aşağıdaki standartlara uygun olması zorunludur.

- Her fonksiyonun başında, fonksiyonun ne iş yaptığını, hangi parametreleri aldığını ve ne döndürdüğünü açıklayan belgelendirme metni bulunmalıdır.
- Model adı ve evrim parametreleri yapılandırma dosyasından okunmalı, kaynak kod içerisine sabit olarak yazılmamalıdır.
- LLM çağrıları sırasında oluşabilecek bağlantı hataları yakalanmalı ve sistem belirli bir bekleme süresinin ardından işlemi yeniden denemelidir.
- Çalışma süresince konsola düzenli aralıklarla ilerleme durumu hakkında bilgi verilmelidir.


## 8. Çalışma Parametreleri

Sistemin ilk çalıştırmada aşağıdaki parametrelerle test edilmesi önerilmektedir. Bu değerler yapılandırma dosyası üzerinden değiştirilebilir olmalıdır.

Popülasyon büyüklüğü on, jenerasyon sayısı beş, mutasyon olasılığı yüzde yirmi, çaprazlama olasılığı yüzde seksen olarak belirlenmiştir. Mini-batch değerlendirmesinde her birey için rastgele seçilecek örnek sayısı yirmi ile sınırlandırılacaktır. Her LLM çağrısı arasına yarım saniyelik zorunlu bekleme süresi eklenecektir.


## 9. Teslimat Kapsamı ve Beklentiler

Coder Agent tarafından teslim edilecek paket aşağıdaki bileşenleri içermelidir.

- Yukarıda tanımlanan modüler yapıya uygun, hatasız çalışan ve bağımlılıkları `requirements.txt` dosyasında listelenmiş kaynak kod.
- Sistemin bir kez çalıştırılmasıyla baştan sona evrim sürecini tamamlayan ve çıktıları `outputs` klasörüne yazan uygulama.
- Çalışma sonunda otomatik olarak oluşturulmuş doğruluk eğrisi grafiği ve nihai rapor dosyası.
- Kod içerisinde yeterli düzeyde açıklama ve belgelendirme.


## 10. Literatür ile İlişkilendirme Notu

Geliştirici kodun işleyişine doğrudan dahil olmayacak olsa da, rapor aşamasında kullanılmak üzere aşağıdaki akademik çalışmalara aşina olunması beklenmektedir.

- **Sprig (Nisan 2026):** Sistem komutlarını genetik algoritma ile optimize eden, düzenleme tabanlı bir yaklaşımdır. Bu proje, Sprig'in basitleştirilmiş bir uyarlaması niteliğindedir.
- **GEPA (2025–2026):** Genetik algoritmaların komut optimizasyonunda pekiştirmeli öğrenme yöntemleriyle rekabet edebileceğini gösteren bir çalışmadır. Bu proje, GEPA'nın temel prensiplerini uygulamalı olarak doğrulamayı amaçlamaktadır.
- **RoboPhD (Nisan 2026):** Büyük dil modelleri rehberliğinde evrimsel ajan optimizasyonunu ele alan ve GEPA ile karşılaştırmalı analizler sunan güncel bir çalışmadır.


## 11. Sonuç ve Yürürlük

Bu yönerge, Coder Agent'ın projeyi eksiksiz ve standartlara uygun biçimde geliştirmesi için gerekli tüm teknik ve işlevsel gereksinimleri tanımlamaktadır. Yönergede belirtilen modüler yapıya, kodlama standartlarına ve teslimat beklentilerine eksiksiz uyulması, projenin akademik değerlendirmede başarılı olması açısından kritik önem taşımaktadır.

Geliştirme sürecinde ortaya çıkabilecek her türlü soru veya belirsizlik için iş analizi birimi ile iletişime geçilmesi önerilir.

**Yönergeyi Hazırlayan:** İş Analizi Birimi  
**Onay Tarihi:** 08 Nisan 2026  
**Geçerlilik Süresi:** 10 Nisan 2026 tarihine kadar