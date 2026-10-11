# yonga nasıl çalışır

yonga, bir Hugging Face depo kimliğini makinenizdeki bir Intel aygıtı için hazırlanmış ve
doğrulanmış bir OpenVINO modeline dönüştürür ve her seçimi neden yaptığını size söyler. Bu sayfa
bunun arkasındaki fikirleri açıklar: hazırlama hattı, aygıtın ve nicemleme (quantization)
profilinin nasıl seçildiği, hazırlanmış modelin (artifact) ne olduğu, yonga'nın uyumluluk
bilgisinin nereden geldiği, hataların nasıl bildirildiği ve yonga'nın bilerek asla yapmadığı
şeyler. [Başlamak](getting-started.md) için bunların hiçbirini bilmeniz gerekmez, ama `inspect` ve
`prepare` çıktılarını okumayı çok kolaylaştırır.

## Hazırlama hattı { #pipeline }

`yonga prepare MODEL` önce bir hazırlama planı yazdırır ve pahalı bir işe başlamadan önce onayınızı
ister. Onayladığınızda yedi aşama çalışır; ilerleme satırları da aynı şekilde numaralanır:

| Aşama | Ne olur |
|---|---|
| 1/7 Inspecting source | Hub üst verisini okur (ağırlıkları değil): biçimler, mimari, parametre sayısı, lisans. Dallar ve etiketler değişmez bir commit'e çözümlenir. |
| 2/7 Resolving compatibility | Mimari için uyumluluk kayıt defteri (registry) kurallarını eşleştirir, aygıtı ve dönüştürme profilini seçer, bilinen onarımları öngörür. |
| 3/7 Preflight | Diski, geçici alanı, belleği, kimlik doğrulamayı, uzak kodu ve dışa aktarıcı paket sürümlerini denetler. Tamamlanamayacak bir iş, hiçbir şey indirilmeden reddedilir. |
| 4/7 Downloading | Yalnızca dönüştürmenin okuduğu dosyaları, sabitlenmiş commit'ten, olağan Hugging Face önbelleğine indirir. |
| 5/7 Converting | Profilin ayarlarıyla optimum-intel ve NNCF kullanarak dışa aktarır ve nicemler; sonuç geçici bir hazırlık (staging) dizinine yazılır. |
| 6/7 Validating / repairing | OpenVINO IR dosyalarının dökümünü çıkarır, tokenizer'ları geri okur ve yalnızca kayıt defterinde listelenmiş dar kapsamlı onarımları uygular. |
| 7/7 Probing runtime | Modeli hedef aygıtta derler ve duman testi olarak birkaç token üretir. |

Hazırlık dizini, ancak her aşama başarılı olduğunda, yapılan her şeyi kaydeden bir manifest ile
birlikte hazır bir artifact olarak yerine taşınır. Başarısız bir iş geride yarım kalmış bir
artifact bırakmaz; en fazla, inceleyebilmeniz için hazırlık dizinini saklar ve yerini söyler.

Referans makinede gerçek bir `prepare` çalıştırmasının plan başlığı (kısaltılmış):

```text
Requested model             Qwen/Qwen3-0.6B@c1899de289a04d12100db370d814…
Resolved source             Qwen/Qwen3-0.6B
Revision                    c1899de289a0
Relationship                original
Source format               safetensors
Architecture                qwen3  rule arch-qwen3
Task                        text-generation-with-past
Device decision             NPU  supported (tested)  NPU was requested
                            explicitly; yonga never switches device on
                            its own.
Profile                     npu-int4-sym-g128
                            NPU group-wise INT4; the NPU default below
                            5B parameters
Quantization                int4, group size 128, ratio 1, symmetric
Chosen because              751.6M parameters (Hugging Face Hub count
                            from safetensors headers) is below 5B:
                            npu-int4-sym-g128 is the NPU profile for
                            that size (evidence: tested).
Expected download           1.4 GiB  only the files the conversion reads
…
Expected repairs
• ov-tokenizer-precision-string-casing
```

Her aşamanın ayrıntısı [model hattı belirtiminde](https://github.com/ooguz/yonga/blob/main/docs/MODEL_PIPELINE.md)
anlatılır. Pratik tarafı için [Model hazırlama](guide/prepare.md) sayfasına bakın.

## Aygıtlar { #devices }

yonga, OpenVINO'nun bir Intel AI PC'de bildirdiği üç aygıtla çalışır: **CPU**, tümleşik **GPU** ve
**NPU**. `prepare`, `--device auto`, `npu`, `gpu` veya `cpu` alır; varsayılan `auto`'dur.

### Otomatik seçim { #auto-selection }

`--device auto` ile yonga, OpenVINO'nun makinenizde bildirdiği aygıtları sıralar:

1. Kayıt defterinin bu model için **unsupported** (desteklenmiyor) olarak işaretlediği aygıtlar
   elenir.
2. Kalanlar, kayıt defterinin ne kadar iyi çalıştıklarını bildiğine göre sıralanır: önce
   `supported`, sonra `experimental`, sonra `unknown`.
3. Eşit derecede desteklenen aygıtlar varsayılan **GPU, NPU, CPU** sırasıyla alınır. Tümleşik GPU en
   geniş graf yelpazesini kabul eder, NPU en seçici olandır, CPU ise son çaredir.
4. Kayıt defteri mimari için `official` veya `tested` kanıtına dayanan bir aygıt öneriyorsa o aygıt
   en öne geçer.

Planın "Device decision" satırı, bu kurallardan hangisinin karar verdiğini söyler. Yerleşik kayıt
defterindeki her aile GPU'yu önerir; bu yüzden `--device auto`, NPU da çalışacak olsa genellikle
GPU için hazırlar. NPU'yu hedeflemek için açıkça isteyin.

Otomatik kipte seçilen aygıt, model onunla uyumlu olmadığı için yoklamayı geçemezse yonga nedenini
açıklar ve bir sonraki aygıtı denemeden önce size sorar (`--yes` bu soruyu evet diye yanıtlar).
Hata makinenin kendisinin bozuk olduğu anlamına geliyorsa, örneğin NPU aygıt düğümüne erişim izni
yoksa ya da bir sürücü başlatılamıyorsa, geri dönüş (fallback) yapmaz: durur, çünkü önce bunun
düzeltilmesi gerekir.

### Açıkça istenen aygıt sessizce değiştirilmez { #explicit-device }

`--device npu` (ya da `gpu` veya `cpu`) bir tercih değil, bir talimattır. O aygıt modeli
çalıştıramıyorsa `prepare` durur ve bunun yerine ne çalıştırmanız gerektiğini söyler; sessizce başka
bir aygıt için hazırlamaz. Kayıt defteri, aygıtın modeli çalıştıramayacağını `official` ya da
`tested` kanıtla zaten biliyorsa `prepare` indirmeden önce reddeder. Her iki durumda da çıkış kodu
50'dir (yukarıdaki gibi makine bozuksa 10).

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu
```

Bir artifact yalnızca gerçekten yoklamasını geçtiği aygıtlar için etiketlenir; yani `yonga list`
çıktısındaki bir "GPU" artifact'i, sizin GPU'nuzda derlenmiş ve metin üretmiş bir artifact'tir.

## Profiller { #profiles }

**Profil**, adlandırılmış bir dönüştürme ayarları kümesidir: ağırlık biçimi, simetrik ya da
asimetrik nicemleme, grup boyutu, 4 bite sıkıştırılan katmanların oranı ve OpenVINO GenAI için
çalışma zamanı özellikleri. Adlar başlıca ayarları açıkça yazar:

| Adın parçası | Anlamı |
|---|---|
| `npu-`, `gpu-`, `cpu-` | Profilin yazıldığı aygıt |
| `int4`, `int8`, `fp16` | Ağırlık biçimi |
| `sym`, `asym` | Simetrik ya da asimetrik nicemleme |
| `g128`, `cw` | 128'lik gruplarla grup bazında ya da kanal bazında (channel-wise) |
| `r08` | Oran 0,8: katmanların yaklaşık beşte biri INT8'de kalır |
| `ref` | Kalite değerlendirmesi için referans profil; hiçbir zaman varsayılan değildir |

`--profile`'ın kabul ettiği tüm profilleri görmek için şunu çalıştırın:

```bash
yonga registry profiles
```

```text
Profile                 Device  Weights                   Size band  Prio
cpu-int4-asym-g128-r08  CPU     int4 asym g128 ratio 0.8  below 5B    110
cpu-int4-asym-g128      CPU     int4 asym g128            —           100
cpu-int8-asym-ref       CPU     int8 asym                 —            10
cpu-fp16-ref            CPU     fp16                      —             5
gpu-int4-asym-g128-r08  GPU     int4 asym g128 ratio 0.8  below 5B    110
gpu-int4-asym-g128      GPU     int4 asym g128            —           100
npu-int4-sym-g128       NPU     int4 sym g128             below 5B    110
npu-int4-sym-cw         NPU     int4 sym channel-wise     —           100
```

Gerçek çıktıda bir özet sütunu da vardır; yalnızca tek bir aygıtı listelemek için `--device npu`
ekleyin.

### Boyut bantları ve profilin seçilmesi { #size-bands }

Her aygıtın **5B parametrenin altındaki** modeller için bir varsayılanı, boyutu bilinmeyenler de
dahil geri kalan her şey için bir başka varsayılanı vardır:

| Aygıt | 5B parametrenin altı | 5B ve üstü ya da bilinmeyen boyut |
|---|---|---|
| CPU | `cpu-int4-asym-g128-r08` | `cpu-int4-asym-g128` |
| GPU | `gpu-int4-asym-g128-r08` | `gpu-int4-asym-g128` |
| NPU | `npu-int4-sym-g128` | `npu-int4-sym-cw` |

Parametre sayısı, Hugging Face Hub'ın safetensors başlıklarından bildirdiği sayıdır (`yonga inspect`
gösterir); bu yüzden gömme (embedding) katmanlarını ve varsa görüntü kodlayıcısını (vision tower) da içerir. Küçük model varsayılanları daha
fazla hassasiyet korur, çünkü referans makinede on iki model üzerinde yapılan ölçümlerde bu, %9-18
daha fazla disk karşılığında çok daha az kalite kaybettirdi. Öne çıkan sonuçlar
[Test edilen donanım ve modeller](hardware.md#measurements) sayfasındadır.

yonga profili şu sırayla seçer:

1. `--profile` ile verdiğiniz profil her zaman kazanır. Asla başka bir profille değiştirilmez; grup
   boyutu modele uymuyorsa `prepare` onu indirmeden önce reddeder ve uyan profilleri söyler.
2. Kayıt defterinde tam bu model ya da mimarisi için bir kuralın adını verdiği profil.
3. Aksi halde, aygıtın boyut bandı eşleşen en yüksek öncelikli profili.

Otomatik seçim, grup boyutu modelin `config.json` içindeki katman genişliklerini bölemeyen grup
bazlı bir profili de atlar. Planın "Chosen because" satırı nedeni her zaman tek cümlede verir;
artifact'in manifest'i de bunu saklar.

## Artifact'ler { #artifacts }

**Artifact**, hazırlanmış modeldir: OpenVINO IR, tokenizer'lar, sohbet şablonu ve manifest ile
doğrulama kayıtlarını içeren bir `.yonga/` dizini. Artifact'ler veri dizininde, varsayılan olarak
`~/.local/share/yonga/artifacts/` altında, depo adını, kaynak revizyonunu ve profili taşıyan bir
klasörde durur:

```text
~/.local/share/yonga/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--npu-int4-sym-g128
```

- **Artifact kimliği.** Her artifact'in `cc24636a2f49356a` gibi 16 karakterlik bir kimliği vardır;
  depodan, tam kaynak commit'inden ve profilden türetilir. Aynı girdiler her zaman aynı kimliği
  üretir. `prepare` bir depo kimliği alır; `chat`, `probe`, `bench`, `serve` ve `remove` ise
  `prepare` ile `yonga list`'in yazdırdığı artifact kimliğini (ya da artifact'in yolunu) alır.
- **Revizyon sabitleme.** Bir dal ya da etiket, herhangi bir şey indirilmeden önce bir commit'e
  çözümlenir ve her dosya o commit'ten gelir. Belirli bir revizyon seçmek için `--revision`
  kullanın.
- **Manifest.** `.yonga/manifest.json`; istenen ve çözümlenen kaynağı, revizyonu, mimari kuralını,
  dışa aktarma komutunu ve ayarlarını, her onarımı öncesi ve sonrası dosya özetleriyle (hash),
  aygıt başına yoklama sonuçlarını, makineyi ve paket sürümlerini ve her dosyanın özetini kaydeder.
  `yonga publish` model kartını bundan oluşturur.
- **Yeniden kullanım.** Aynı revizyon ve profil için hazır bir artifact zaten varsa `prepare` onu
  yeniden dönüştürmeden ve soru sormadan kullanır. Yeniden hazırlamak için `--force` verin.
  Değişen bir varsayılan profil, eskisinin yanında yeni bir artifact üretir; asla sessizce eskisini
  yeniden kullanmaz.

Kaynak modeliniz Hugging Face önbelleğinde kalır. yonga onu orada okur ve yalnızca kendi
dizinlerine yazar. Artifact silme ve önbellek temizleme [Bakım](guide/maintain.md) sayfasında
anlatılır.

## Uyumluluk kayıt defteri { #registry }

Hangi mimarinin hangi aygıtta, hangi dışa aktarıcı sürümleriyle çalıştığı, hangi hataların
bilindiği ve hangi onarımların güvenli olduğu bilgisi yonga'nın kodundan daha hızlı değişir. Bu
yüzden bu bilgi **uyumluluk kayıt defterinde** durur: kod değil, YAML verisi. Yerleşik
kurallar yonga ile birlikte gelir; mimarileri, profilleri, bilinen sorunları, onarımları,
bildirilmiş türev kaynakları, sohbet davranışını ve GGUF içe aktarma desteğini tanımlar.

Kayıt defterindeki her aygıt iddiası bir **kanıt düzeyi** taşır:

| Kanıt | Anlamı |
|---|---|
| `official` | OpenVINO tarafından, belgelerinde ya da yayımladığı bir model kartında belgelenmiş |
| `tested` | Gerçek donanımda çalıştırılmış; [Test edilen donanım ve modeller](hardware.md) sayfasındaki aileler bu çalıştırmalar sayesinde `tested` düzeyindedir |
| `reported` | `tested`'dan zayıf, `inferred`'dan güçlü |
| `inferred` | İlgili bir aileden çıkarılmış; örneğin bir mistral modelini genel bir kural yakalar ve bu kural CPU ile GPU'yu `supported (inferred)`, NPU'yu `unknown` olarak değerlendirir |
| `unknown` | Güvenilir bilgi yok; kararı yoklama verir |

`yonga inspect MODEL`, aygıt başına kararı kanıtıyla birlikte gösterir. Bir modelle tam olarak
hangi kuralların eşleştiğini, hangilerinin kaybettiğini ve nedenini görmek için şunu çalıştırın:

```bash
yonga registry explain Qwen/Qwen3-0.6B
```

```text
Device compatibility
✓  CPU  supported (tested)  arch-qwen3
✓  GPU  supported (tested)  arch-qwen3
✓  NPU  supported (tested)  arch-qwen3
```

(Bu, mimari, profil, sohbet profili ve türev kaynaklar için aday kuralları listeleyen çok daha uzun
bir çıktının son bölümüdür.)

### Katmanlar ve imzalı uzak paketler { #remote-bundles }

Kayıt defteri katmanlar halinde yüklenir; her katman bir öncekinin kuralını kimliğiyle
değiştirebilir:

1. yonga ile birlikte gelen yerleşik kurallar;
2. `yonga registry update` ile kurulan, isteğe bağlı **imzalı uzak paket**;
3. `~/.config/yonga/registry-overrides/` içindeki kendi geçersiz kılmalarınız.

Her katmanın nerede durduğunu `yonga registry paths` gösterir. Siz söylemedikçe yonga hiçbir kayıt
defteri sunucusuna bağlanmaz: `registry update`, `--url` (ya da yapılandırmada bir `registry_url`)
veya `--file` ile diskte duran bir paket ister. Bir paket ancak `~/.config/yonga/registry-keys/`
dizinine kendi elinizle koyduğunuz bir anahtarla imzalanmışsa, süresi dolmamışsa, kurulu olandan
eski değilse ve yalnızca düz veri içeriyorsa kurulur. Paket, yonga'nın içinde zaten bulunan bir
onarımı ya da araç çağırma ayrıştırıcısını seçebilir, ama asla kod getiremez. Doğrulaması artık
geçmeyen bir paket yok sayılır ve `yonga registry status` nedenini söyler.

Her kural türü ve çözümleme sırası
[kayıt defteri belirtiminde](https://github.com/ooguz/yonga/blob/main/docs/REGISTRY_SPEC.md)
belgelenmiştir.

## Tanılama ve çıkış kodları { #diagnostics }

Bir şey başarısız olduğunda yonga önünüze ham bir derleyici hata dökümü (traceback) koymaz. Hata metni kayıt
defterinin sorun kurallarından geçer ve şunları içeren bir tanılama alırsınız:

- `npu.compiler.unsupported_graph.qwen35` gibi değişmeyen bir **kod**;
- "bu model bu aygıta uymuyor" ile "kurulumunuz bozuk" durumlarını ayıran bir **kategori**;
- kopyalayabileceğiniz gerçek komutlarla **çözüm adımları**;
- ham metni saklayan **iş günlüğünün** yolu. `yonga logs JOB_ID` onu gösterir; `--debug` ise ham
  hata metnini hatanın kendisine ekler.

Her komut, neyin ters gittiğine göre gruplanmış, değişmeyen bir çıkış koduyla biter:

| Kodlar | Anlamı |
|---|---|
| 0, 2 | Başarılı; 2, uyarılarla başarılı (ya da sonuçsuz bir yoklama) demektir |
| 10 | Ortam kullanılamaz: eksik bir paket, sürücü ya da izin |
| 20, 21, 22 | Model kaynağı çözümlenemedi, kimlik doğrulama gerekiyor ya da uzak kod onayı gerekiyor |
| 30 | Yetersiz disk ya da bellek |
| 40, 41, 42 | Dönüştürme, doğrulama ya da gerekli olduğu halde devre dışı bırakılmış bir onarım |
| 50, 51, 52 | Aygıt desteklenmiyor, derleme başarısız ya da metin üretimi başarısız |
| 60 | İptal edildi ya da bir onay reddedildi |
| 70 | Kayıt defteri ya da yapılandırma hatası veya yonga'nın ayrıştıramadığı bir komut satırı |
| 99 | Beklenmeyen bir iç hata |

Her birinin listesi [Çıkış kodları](reference/exit-codes.md) sayfasında, gerçekten karşılaşılmış
hataların adım adım çözümü ise [Sorun giderme](troubleshooting.md) sayfasındadır.

## yonga'nın yapmayacakları { #boundaries }

Bazı sınırlar tasarımın bir parçasıdır ve hiçbir bayrak bunları kapatmaz:

- **Telemetri yok.** yonga, onu nasıl kullandığınıza dair hiçbir şey bildirmez; başlattığı her
  Python sürecinde OpenVINO'nun kendi kullanım raporlamasının yüklenmesini de engeller. Bkz.
  [Gizlilik ve güvenlik](privacy.md).
- **`sudo` yok, sistem değişikliği yok.** yonga asla `sudo`, `apt`, `modprobe` ya da `usermod`
  çalıştırmaz ve kendisini kurduğunuz Python ortamında asla paket kurmaz, yükseltmez ya da
  kaldırmaz. Makinede bir şeyin değişmesi gerekiyorsa çalıştırmanız için komutu yazdırır. `pip`'i
  çalıştırdığı tek yer, kendi önbellek dizini altındaki yönetilen dışa aktarma ortamıdır ve bunu
  da siz planda onayladıktan sonra yapar.
- **Hugging Face önbelleğinize dokunmaz.** Kaynak modeller önbellekten okunur ve asla yerinde
  değiştirilmez. Onarımlar, dışa aktarılan dosyaların yonga'ya ait kopyasına uygulanır.
- **İzin vermedikçe uzak kod yok.** Yalnızca kendi Python kodu çalıştırılarak dönüştürülebilen bir
  depo için `--trust-remote-code` gerekir. `--yes` bunu asla onaylamaz; bayrak olmadan `prepare` 22
  çıkış koduyla durur.
- **Sessiz değişiklik yok.** yonga ilgisiz bir modeli asla araya sokmaz, açıkça istenen bir aygıtı
  asla değiştirmez ve adını verdiğiniz bir profilin yerine başkasını koymaz. Bildirilmiş bir türev
  kaynak yalnızca önerilir ve onayınızı gerektirir.
