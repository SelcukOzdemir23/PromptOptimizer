Tamamdır. Veri seti seçimi, bu projenin iki gün içinde başarıyla tamamlanması için en kritik kararlardan biri. Adayları belirledim ve senin için üç ana başlık altında karşılaştırmalı olarak değerlendireceğim.

---

### 🎯 Önerilen Veri Setleri ve Analizi

Projenin başarısı için bir veri setinde aradığımız temel özellikler şunlardır: Sınıfların net bir şekilde ayrılabilir olması, veri setinin küçük ve hızlı işlenebilir olması, etiketlerin doğru ve güvenilir olması.

Bu kriterler ışığında öne çıkan adayları şöyle sıralayabiliriz:

**1. AG News (İngilizce) – Resmî Bir Benchmark**
*   **Veri Seti:** Haber başlıklarını "Dünya", "Spor", "İş Dünyası" ve "Bilim/Teknoloji" olmak üzere dört kategoriye ayırır..
*   **Boyut ve Hız:** Oldukça büyüktür ancak biz sadece `test.csv` dosyasındaki 7.600 örnek üzerinde çalışacağız. Bu, hem hızlı hem de sonuçları genellenebilir kılmak için yeterlidir.
*   **Avantajları:** Dünya çapında birçok akademik çalışmada ve prompt optimizasyonu makalesinde (ör. GEPA, Sprig) standart olarak kullanılır. Raporunda "Bu veri seti, literatürde yaygın olarak kullanılan bir ölçüttür" diyebilmek, çalışmanın ciddiyetini anında artırır..
*   **Zorlukları:** İngilizce bir veri setidir ve LLM'in haber başlıklarını sınıflandırması, duygu analizine göre biraz daha fazla "dünya bilgisi" gerektirebilir. Bu, prompt'un farkını ortaya çıkarmak için iyi bir testtir.

**2. Türkçe Haber Manşetleri (7 Sınıf) – Hızlı ve Temiz Bir Seçenek**
*   **Veri Seti:** Ekonomi, siyaset, yaşam, teknoloji, magazin, sağlık ve spor olmak üzere 7 farklı kategoride toplam 4.200 haber başlığı içerir..
*   **Boyut ve Hız:** Boyutu (4.200 örnek) idealdir. 7 sınıflı olması, problemi rastgele tahmin başarısının (~%14) oldukça üzerine çıkarılabilir kılar ve genetik algoritmanın yaptığı iyileştirmeyi net bir şekilde görmeni sağlar.
*   **Avantajları:** Türkçe olması, eğer raporunda veya sunumunda yerelleştirme vurgusu yapmak istersen büyük bir artıdır. Ayrıca, veri seti doğrudan Kaggle'da "Anıl 1055" kullanıcısı tarafından temiz bir şekilde paylaşılmıştır..

**3. Türkçe Duygu Analizi – En Basit ve Etkili Problem**
*   **Veri Seti:** "Pozitif" ve "Negatif" olarak etiketlenmiş Türkçe haber metinlerinden oluşur. Sınıf dengesi sağlanmıştır..
*   **Boyut ve Hız:** "Sizzu" takımı tarafından paylaşılan bu versiyon, doğrudan kullanıma hazırdır ve boyutu da oldukça uygundur..
*   **Avantajları:** İkili sınıflandırma (pozitif/negatif), LLM'lerin en başarılı olduğu ve prompt mühendisliğinin etkisinin en hızlı gözlemlenebildiği problemdir. Başlangıçta %60-70 civarında bir doğrulukla başlayıp, evrim sonunda %80-85 bandına çıkmak çok olasıdır. Bu net sıçrama, raporunda "evrimin gücünü" göstermek için harika bir grafiktir.
*   **Dezavantajı:** Problem nispeten daha basit olduğu için, manuel yazılan iyi bir prompt ile evrimleşmiş prompt arasındaki fark (%5-10) çok çarpıcı olmayabilir. Yine de evrimin optimizasyon yeteneğini göstermek için fazlasıyla yeterlidir.

---

### 🏆 Proje İçin Nihai Tavsiye

Projenin "iki günde bitme" ve "hesapsal zekanın gücünü gösterme" hedeflerini en iyi dengeleyen veri seti **AG News (İngilizce)** 'dir.

Gerekçelerim şunlardır:
*   **Akademik Ağırlık:** Bu veri seti, senin kullandığın yöntemin (LLM ile Neuroevolution) tam da test edildiği bir "altın standarttır". Raporunun literatür kısmında "AG News üzerinde yapılan çalışmalar..." diye başlayan bir cümle, jüriye projenin temellerinin sağlam olduğunu hissettirir..
*   **Tekrar Kullanılabilirlik:** Bu projeyi ileride bir makaleye veya daha büyük bir projeye dönüştürmek istersen, sonuçlarını uluslararası literatürdeki diğer çalışmalarla doğrudan karşılaştırabilirsin.
*   **Uygulama Kolaylığı:** Kaggle üzerinde `amananandrai/ag-news-classification-dataset` adresinde hazır olarak bulunur ve doğrudan `pandas` ile okunabilir..

### ⚙️ Veri Setini Projeye Entegre Etme Planı

Bu veri setini projene şu şekilde dahil edeceğiz:
1.  **Kaynak:** Kaggle'dan "AG News Classification Dataset" indirilecek.
2.  **Kullanılacak Bölüm:** Yalnızca `test.csv` dosyasındaki 7.600 örnek kullanılacak. Bu, hem API maliyetini düşürür hem de süreci hızlandırır.
3.  **Veri Yapısı:** Dosyada üç sütun bulunur: `Class Index`, `Title` ve `Description`. Biz sadece `Title` (haber başlığı) sütununu LLM'e girdi olarak vereceğiz.
4.  **Etiketler:** `Class Index` sütunu 1'den 4'e kadar sayısal değerler içerir. Bunları "World", "Sports", "Business", "Sci/Tech" olarak eşleştireceğiz.

Bu veri seti üzerinde çalışmak, projeni sadece bir ödev olmaktan çıkarıp, tekrarlanabilir ve uluslararası standartlarda bir araştırma prototipine dönüştürecektir.

Şimdi sırada ne var? İstersen bu veri setini okuyup kullanıma hazır hale getirecek `data_loader.py` modülünü birlikte yazmaya başlayabiliriz.