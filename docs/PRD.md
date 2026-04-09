Anlaşıldı. Aşağıda, doğal bir dille ancak **resmî ve teknik bir doküman dili** kullanılarak hazırlanmış, iki gün içinde tamamlanabilecek şekilde sadeleştirilmiş bir Proje Gereksinim Dokümanı bulunmaktadır. İçerik tamamen Markdown formatındadır ve herhangi bir kod bloğu ya da terminal komutu içermez.

---

# EvoPrompt Optimizer – Proje Uygulama Dokümanı
**Ders:** Hesapsal Zeka – Vize Ödevi  
**Tarih:** 08 Nisan 2026  
**Süre:** 2 Günlük Uygulama Planı  
**Amaç:** Genetik Algoritma ile Büyük Dil Modeli (LLM) Prompt Optimizasyonu

---

## 1. Projenin Tanımı ve Kapsamı

Bu proje, evrimsel algoritmalar ile büyük dil modellerini birleştiren hibrit bir sistemin uygulamalı olarak hayata geçirilmesini amaçlamaktadır. Çalışmanın merkezinde, belirli bir metin sınıflandırma problemi için en yüksek doğruluğu sağlayan doğal dil komutunun (prompt) genetik algoritma yardımıyla otomatik olarak keşfedilmesi yer almaktadır. Sistem, büyük dil modelinin parametrelerine müdahale etmemekte; yalnızca modele iletilen yönergeyi evrimsel süreçlerle iyileştirmektedir.

Uygulama, iki iş günü içinde tamamlanabilecek şekilde kurgulanmıştır. Kullanılacak veri kümesi küçük ölçekli olacak, genetik algoritma parametreleri ise hızlı sonuç alınabilecek biçimde ayarlanacaktır.

---

## 2. Kullanılacak Güncel Teknolojiler ve Gerekçeleri

Proje, 2025 ve 2026 yıllarında yaygın olarak benimsenen araçlar ve kütüphaneler üzerine inşa edilecektir.

- **Ollama (llama3.1:8b):** Yerel çalışan, API anahtarı gerektirmeyen, hızlı ve sınırsız inference. Kota sorunu yoktur.
- **DEAP Kütüphanesi:** Genetik algoritma işlemlerinin standart bir çerçeve içinde yürütülmesini sağlayan, akademik çalışmalarda sıklıkla tercih edilen bir Python paketidir.
- **NLTK WordNet:** Mutasyon aşamasında anlam bütünlüğünü korumak amacıyla eş anlamlı sözcük değişimleri için kullanılacaktır.
- **Matplotlib ve Pandas:** Evrim sürecinin görselleştirilmesi ve sayısal çıktıların düzenlenmesinde görev alacaktır.

---

## 3. Modüler Sistem Mimarisi

Uygulama, yazılım mühendisliği prensiplerine uygun olarak dört bağımsız modül etrafında yapılandırılacaktır. Bu modüller birbirleriyle yalnızca belirli arayüzler üzerinden iletişim kuracaktır.

### Modül 1: Veri Yönetim Modülü

Sorumluluğu, ham veri kümesini yüklemek ve genetik algoritmanın kullanımına uygun hale getirmektir. Bu modül aşağıdaki işlevleri yerine getirecektir:

- Ham veri kümesinin diskten okunması.
- Veri kümesinin evrim aşaması için kullanılacak eğitim havuzu ile nihai değerlendirme için kullanılacak test kümesi olarak iki parçaya ayrılması.
- Metin etiketlerinin sayısal değerlere dönüştürülmesi ve sınıf isimlerinin kayıt altına alınması.

### Modül 2: Büyük Dil Modeli Arayüz Modülü

Bu modül, sistemin geri kalanı ile bulut tabanlı dil modeli API'leri arasındaki tüm iletişimi yönetmektedir. Görev tanımı şu şekildedir:

- Verilen bir metin ve bir sistem komutu (prompt) için modelden tahmin talep etmek.
- Modelden dönen yanıtı düzenli ifadeler veya biçim denetimi ile temizleyerek sayısal sınıf etiketine dönüştürmek.
- Ağ gecikmeleri, kota aşımları veya geçici hatalar karşısında sistemi korumak amacıyla üstel geri çekilme (exponential backoff) mekanizması uygulamak.

### Modül 3: Evrimsel Algoritma Motoru

Genetik algoritmanın tüm temel operasyonları bu modülde DEAP kütüphanesi aracılığıyla gerçekleştirilecektir. Operasyonel detaylar aşağıda sıralanmıştır:

- **Popülasyon Başlatma:** İlk nesil tamamen rastgele dizgilerden oluşturulmayacaktır. Popülasyona, insan tarafından yazılmış makul bir başlangıç komutu (seed prompt) ve bu komutun WordNet eş anlamlıları kullanılarak türetilmiş varyasyonları yerleştirilecektir.
- **Fitness Değerlendirmesi:** Her birey (prompt metni) için doğruluk puanı, eğitim havuzundan rastgele seçilen küçük bir örneklem (mini-batch) üzerinden hesaplanacaktır. Bu yaklaşım, her birey için tüm veri kümesinin sorgulanmasından kaynaklanacak zaman ve mali yükü ortadan kaldırmaktadır.
- **Seçilim Yöntemi:** Turnuva seçimi (tournament selection) uygulanacaktır. Turnuva boyutu küçük tutulacaktır.
- **Çaprazlama Yöntemi:** İki ebeveyn komut metninin belirli bir kesme noktasından birleştirilmesiyle yeni bireyler üretilecektir.
- **Mutasyon Yöntemi:** Düşük bir olasılıkla, komut metni içerisindeki bir sözcüğün WordNet'ten alınan bir eş anlamlısı ile değiştirilmesi şeklinde gerçekleşecektir. Bu yöntem, komutun dilbilgisel yapısını büyük ölçüde korumayı hedeflemektedir.

### Modül 4: Görselleştirme ve Raporlama Modülü

Deney sonuçlarının akademik sunuma uygun biçimde kaydedilmesinden sorumludur. Çalışma tamamlandığında aşağıdaki çıktıları otomatik olarak üretecektir:

- Jenerasyonlar boyunca ortalama ve en yüksek doğruluk değerlerinin değişimini gösteren bir çizgi grafiği.
- Her jenerasyonda elde edilen en iyi komut metnini ve parametreleri içeren bir kayıt dosyası.
- Nihai test kümesi üzerinde ulaşılan başarı oranını ve en iyi komutun tam metnini içeren özet bir rapor dosyası.

---

## 4. Deneysel İş Akışı ve Zaman Çizelgesi

Projenin iki gün içinde tamamlanabilmesi için aşağıdaki aşamalı plan önerilmektedir.

**Birinci Gün – Altyapı ve Modül Geliştirme**

- İlk olarak Veri Yönetim Modülü ve Büyük Dil Modeli Arayüz Modülü yazılacak ve birbirinden bağımsız olarak test edilecektir.
- Ardından DEAP kütüphanesi kullanılarak Evrimsel Algoritma Motoru'nun iskeleti oluşturulacaktır. Başlangıç popülasyonunun üretilmesi ve fitness fonksiyonunun LLM modülüne bağlanması bu aşamada tamamlanacaktır.
- Gün sonunda sistemin, yalnızca bir jenerasyon için de olsa, baştan sona hatasız çalıştığı doğrulanacaktır.

**İkinci Gün – Evrim Koşumu ve Raporlama**

- Evrimsel algoritma parametreleri (popülasyon büyüklüğü, jenerasyon sayısı) nihai değerlerine ayarlanacak ve tam sürüm çalıştırma başlatılacaktır.
- Çalışma sırasında Görselleştirme ve Raporlama Modülü devreye alınacak ve çıktılar kaydedilecektir.
- Süreç tamamlandığında, elde edilen en iyi komut, daha önce hiç kullanılmamış olan test kümesi üzerinde değerlendirilecektir.
- Sonuçlar yorumlanarak proje raporunun ilgili bölümleri yazılacaktır.

---

## 5. Literatür ile İlişkilendirme

Projenin teorik dayanağı olarak, 2025 ve 2026 yıllarında yayımlanmış olan ve prompt optimizasyonunda evrimsel yaklaşımların etkinliğini gösteren çalışmalara atıfta bulunulacaktır. Bu bağlamda, genetik algoritmaların büyük dil modelleri için komut türetme sürecinde sağladığı esneklik ve keşif kabiliyeti, geleneksel pekiştirmeli öğrenme yöntemleri ile karşılaştırmalı olarak ele alınacaktır. Proje raporunda, bu yöntemin özellikle düşük kaynak tüketimi ile yüksek performans sunma potansiyeli vurgulanacaktır.

---

## 6. Değerlendirme Kriterleri ve Başarı Göstergeleri

Çalışmanın başarısı aşağıdaki ölçütler üzerinden değerlendirilecektir:

- Genetik algoritma sonucunda elde edilen komutun doğruluğu, insan eliyle yazılmış başlangıç komutunun doğruluğundan sayısal olarak daha yüksek olmalıdır.
- Jenerasyon ilerledikçe popülasyonun ortalama doğruluğunda istikrarlı bir artış eğilimi gözlemlenmelidir.
- Üretilen görsel çıktılar (grafik) ve metin dosyaları, deneyin yeniden üretilebilirliğini ve sonuçların yorumlanabilirliğini destekleyecek nitelikte olmalıdır.
- Tüm kod modülleri, harici bir kullanıcı tarafından kolaylıkla çalıştırılabilecek ve konfigürasyon değişikliklerine izin verecek şekilde düzenlenmelidir.

---

## 7. Risk Değerlendirmesi ve Önlemler

**Risk:** LLM'nin beklenmeyen veya sınıflar dışında yanıtlar üretmesi (örn. "Politics", "Entertainment").
**Önlem:** Yanıt işleme katmanı, yalnızca geçerli sınıf isimlerini kabul edecek, eşleşmeyen yanıtlar `None` olarak işaretlenecektir.

**Risk:** Evrim sürecinin eğitim verisine aşırı uyum sağlaması (overfitting).
**Önlem:** Nihai başarı ölçümü, evrim sürecinde hiç kullanılmamış, bağımsız bir test kümesi üzerinden yapılacak ve raporda her iki doğruluk değeri ayrı ayrı raporlanacaktır.

---

Bu doküman, projenin uygulama aşamasında referans alınacak temel gereksinimleri ve mimari kararları ortaya koymaktadır. Uygulamanın belirtilen modüler yapıya sadık kalarak geliştirilmesi, hem kodun sürdürülebilirliğini sağlayacak hem de akademik değerlendirmede yöntemsel netliği artıracaktır.