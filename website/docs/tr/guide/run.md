# Sohbet, yoklama, ölçüm ve değerlendirme

Bir modeli hazırladıktan sonra onu makinenizde dört komutla kullanırsınız: `chat` terminalde modelle konuşur, `probe` modelin bir aygıtta derlenip çalıştığını yoklar, `bench` hızını ölçer, `eval` ise nicemlemenin (quantization) yanıtları ne kadar değiştirdiğini ölçer. Beşinci komut `publish`, modeli Hugging Face Hub'a yükler. Bunların hiçbiri model indirmez veya dönüştürmez; henüz bir model hazırlamadıysanız [Model hazırlama](prepare.md) sayfasından başlayın.

## Artifact'i seçmek {#choose-artifact}

Bu sayfadaki her komut aynı türden bir başvuru alır: `yonga list` çıktısındaki hazırlanmış model (artifact) kimliği, artifact'in dizini ya da onu hazırlarken kullandığınız Hugging Face model kimliği.

```bash
yonga list
yonga chat cc24636a2f49356a
yonga chat Qwen/Qwen3-0.6B
```

Bir model kimliği çoğu zaman birden fazla artifact'e karşılık gelir; örneğin biri NPU, biri GPU için hazırlanmış olabilir. yonga bu durumda hazır olanların en yenisini alır ve hangisini seçtiğini söyler:

```text
Using ID (PROFILE, DEVICE), the newest of N ready artifacts for MODEL; pass the artifact id to choose another.
```

Belirli bir artifact istiyorsanız kimliğini verin. Bu makinede hiç hazırlanmamış bir model kimliği, önce çalıştırmanız gereken `yonga prepare` komutuyla birlikte reddedilir; yan etki olarak hiçbir şey dönüştürülmez.

## Terminalde sohbet {#chat}

Hazırlanmış bir modelle sohbet etmek için şunu çalıştırın:

```bash
yonga chat Qwen/Qwen3-0.6B
```

yonga önce modeli yükler (derler), ardından model, aygıt, profil, çıktı sınırı, örnekleme ayarları ve derleme önbelleğini gösteren bir açılış paneli çizer. Seçilen aygıtta yüklenemeyen bir model `chat` komutunu orada, nedenini söyleyen bir [çıkış koduyla](../reference/exit-codes.md) sonlandırır; örneğin NPU derleyicisi modeli reddettiğinde 51.

Aşağıdaki, referans makinenin NPU'sunda Qwen3-0.6B ile yapılmış gerçek bir oturumdur (kısaltılmıştır):

```text
Loading the model on NPU...
╭──────────────────────────────── Local Chat ────────────────────────────────╮
│  Model              cc24636a2f49356a                                       │
│  Source             Qwen/Qwen3-0.6B@c1899de289a0                           │
│  Profile            npu-int4-sym-g128                                      │
│  Device             NPU                                                    │
│  Max output         4096 tokens                                            │
│  Reasoning display  hidden                                                 │
│  Compile cache      enabled                                                │
│  …                                                                         │
│  Chat profile       chat-qwen3                                             │
│  Thinking mode      off (enable_thinking=false)                            │
│  Sampling           temp 0.7 • top-p 0.8 • top-k 20                        │
│  Metrics footer     on                                                     │
╰──────────────────────────── /help for commands ────────────────────────────╯
Generating a reply...
╭────────────────────────────────────────────────────────────────────────────╮
│  An NPU is a term used in the field of computing and networking,           │
│  referring to a type of network protocol or data format.                   │
╰────────────────────────────────────────────────────────────────────────────╯
TTFT 391 ms • 44.3 tok/s • TPOT 22.6 ms/token • 22 → 27 tokens
Goodbye.
```

Yanıt yanlış; INT4'e nicemlenmiş 0.6B'lik bir model için bu olağan dışı değil. Nicemlemenin bedelini sayıya dökmek için [eval](#eval) bölümüne bakın.

### Seçenekler {#chat-options}

| Seçenek | Ne yapar |
|---|---|
| `--device auto\|npu\|gpu\|cpu` | Modelin çalışacağı aygıt. Verilmezse yonga önce ayarlarınızdaki varsayılan aygıtı, ardından artifact'in hazırlandığı ve yoklandığı aygıtı kullanır. |
| `--max-new-tokens N` | Yanıt başına çıktı token sınırı. Varsayılan değer modelin sohbet profilinden gelir (yukarıdaki oturumda 4096). |
| `--temperature X` | Örnekleme sıcaklığı. `0` açgözlü (greedy) kod çözme demektir. |
| `--top-p X` | Çekirdek (nucleus) örnekleme olasılığı. |
| `--top-k N` | Top-k örnekleme sınırı. |
| `--system TEXT` | Oturumun tamamı için sistem istemi. |
| `--show-reasoning` | Modelin akıl yürütme bloklarını gizlemek yerine gösterir (aşağıya bakın). |

Örnekleme varsayılanları, uyumluluk kayıt defterindeki (registry) sohbet profilinden gelir ve açılış panelinde görünür. `chat`, `--json` kabul etmez: bir sohbet tek bir belge değildir. Makinenin okuyabileceği çıktı için `yonga bench --json` kullanın.

### Sohbet içindeki komutlar {#chat-commands}

Mesaj göndermek yerine komut çalıştırmak için satıra `/` ile başlayın. `/help` bu listeyi yazdırır.

| Komut | Ne yapar |
|---|---|
| `/help` | Komut listesini gösterir. |
| `/reset` | Sohbet geçmişini unutur (sistem istemi korunur). |
| `/clear` | Ekranı temizler. |
| `/tokens N` | Bir yanıttaki en fazla token sayısını ayarlar. |
| `/temp X` | Örnekleme sıcaklığını 0.0 ile 2.0 arasında ayarlar (`/temperature` da çalışır). |
| `/top-p X` | Çekirdek örneklemeyi 0.0 ile 1.0 arasında ayarlar. |
| `/top-k N` | Top-k örneklemeyi ayarlar; `0` kapatır. |
| `/reasoning on\|off` | Modelin akıl yürütme bloğunu gösterir veya gizler. |
| `/stats on\|off` | Ölçüm satırını gösterir veya gizler. |
| `/model` | Yüklü modeli, aygıtı ve profili yeniden gösterir. |
| `/exit` | Sohbetten çıkar (`/quit`, `/q` ve `/bye` da çalışır). |

### Akıl yürütme ve düşünme modu {#reasoning}

Bazı modeller (örneğin Qwen3) yanıttan önce gizli bir akıl yürütme bloğu yazar. yonga bu bloğu varsayılan olarak çıkarır ve yalnızca yanıtı gösterir. `--show-reasoning` ile akıl yürütme, etiketleri olmadan, yanıtın üstünde soluk bir panelde görünür. Sohbet şablonunda düşünme anahtarı olan bir modelde `--show-reasoning`, oturum için düşünme modunu da açar.

`/reasoning on` yalnızca neyin gösterileceğini değiştirir. Oturumda düşünme modu kapalıysa model gösterilecek bir akıl yürütme üretmeyebilir; yonga bunu söyler ve sohbeti `--show-reasoning` ile yeniden başlatmanızı önerir.

### Ölçüm satırı {#metrics}

`chat` her yanıttan sonra şuna benzer soluk bir satır yazdırır:

```text
TTFT 391 ms • 44.3 tok/s • TPOT 22.6 ms/token • 22 → 27 tokens
```

Bunlar sırasıyla ilk token süresi, üretim hızı, çıktı token'ı başına süre ve token cinsinden istem ile yanıt uzunluklarıdır. `/stats off` satırı gizler. Terimler [bench](#bench) bölümünde açıklanıyor.

### Yanıtı durdurmak, çıkmak {#stopping}

- **Yanıt üretilirken Ctrl-C** üretimi bir sonraki token'da durdurur, yanıtı ve onu isteyen mesajı birlikte atar ve sizi giriş satırına geri döndürür. Üretim durmadan basılan ikinci Ctrl-C, `chat` komutunu sonlandırır (çıkış kodu 60).
- **Giriş satırındayken Ctrl-C veya Ctrl-D**, `/exit` ya da yönlendirilmiş girdinin bitmesi sohbetten çıkarır.

Son mesaj başarısız olduysa ve hemen ardından çıkarsanız, `chat` 0 yerine o hatanın koduyla çıkar; böylece bir betik bunu fark edebilir.

!!! tip "İpucu"
    `chat` standart girdiyi okur, bu yüzden hızlı bir denetimi betiğe dökebilirsiniz:

    ```bash
    printf 'What is 2+2? Answer briefly.\n/exit\n' | yonga chat Qwen/Qwen3-0.6B
    ```

!!! note "Not"
    Giriş satırına yazdıklarınız, ok tuşlarıyla geri çağırabilmeniz için `~/.local/share/yonga/chat-history/` altında istem geçmişi olarak tutulur. Yanıtlar saklanmaz. Bkz. [Gizlilik ve güvenlik](../privacy.md).

## probe ile aygıtı yoklamak {#probe}

`probe`, hazırlanmış bir artifact'i bir aygıtta derler ve neyin çalıştığını raporlar. Tokenizer'ı okur, dil modelini derler ve ham derlemenin tek başına karar veremediği yerlerde gerçek OpenVINO GenAI hattını (pipeline) kurup birkaç token üretir. NPU'da bu her LLM için geçerlidir.

```bash
yonga probe Qwen/Qwen3-0.6B --device npu
```

```text
NPU  ✓ ok
·  language    deferred  579 ms  the raw compile is not conclusive here; the GenAI pipeline decides
✓  tokenizer   ok  28 ms
✓  smoke test  ok  2.0 s  generated 24 token(s)
```

Varsayılan `--device auto`'dur ve aday olan her aygıtı dener. Bir aygıt başarısız olurken bir diğeri geçerse, başarısız olan yalnızca bilgi amaçlı raporlanır. `--all-submodels` yalnızca dil modelini değil, her alt modeli derler. `--json` raporu JSON olarak verir.

### Büyük modeller: --quick ve --timeout {#probe-quick-timeout}

Küçük bir modelde hattın kurulması birkaç saniye sürer. NPU'daki büyük bir modelde ise bu tam bir NPU derlemesidir; on dakikayı rahatça aşabilir ve birkaç GiB RAM kullanabilir. İki seçenek bu süreyi kısaltır; birlikte kullanılamazlar:

```bash
yonga probe ARTIFACT --device npu --quick         # hattı hiç kurmaz
yonga probe ARTIFACT --device npu --timeout 600   # hattı alt süreçte kurar, 600 sn sonra durdurur
```

İkisi de aygıtı **inconclusive** (sonuçsuz) bırakabilir: ne çalışıyor ne de başarısız sayılır. Hiçbir aygıt geçmediğinde ve hiçbiri kesin olarak başarısız olmadığında `probe` **2** koduyla çıkar (`--json` çıktısında `"inconclusive": true`):

```text
NPU  ! inconclusive
·  language   deferred  643 ms  the raw compile is not conclusive here; the GenAI pipeline decides
✓  tokenizer  ok  27 ms
…
  - Settle it by building the pipeline: yonga probe … --device npu
  - Or bound the wait: yonga probe … --device npu --timeout SECONDS
```

Sonucu netleştirmek için komutu `--quick` olmadan ya da daha uzun bir `--timeout` ile yeniden çalıştırın.

| Çıkış kodu | Anlamı |
|---|---|
| 0 | En az bir aygıt geçti. |
| 2 | Sonuçsuz: hiçbir aygıt geçmedi, hiçbiri de kesin olarak başarısız olmadı. |
| 51 ve diğerleri | İlk kesin hatanın kodu; örneğin NPU derleyicisi modeli reddettiğinde 51. |

`probe` sizin yerinize aygıt değiştirmez. Ayrıntılar [probe belirtiminde](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#11-probe).

## bench ile hız ölçmek {#bench}

`bench` sabit bir istem kümesini çalıştırır, yeniden üretilebilir bir JSON sonucu kaydeder ve bir özet yazdırır.

```bash
yonga bench Qwen/Qwen3-0.6B --device npu
```

| Seçenek | Varsayılan | Ne yapar |
|---|---|---|
| `--suite` | `standard` | `quick` (tek kısa istem), `standard` (kısa, orta ve uzun bağlamlı istemler) veya `long` (buna daha uzun bir çözümleme istemi ekler). |
| `--runs N` | 3 | İstem başına ölçülen çalıştırma sayısı. |
| `--warmups N` | 1 | İstem başına ısınma çalıştırması. Kaydedilir ama ortalamalara katılmaz. |
| `--device` | artifact'in aygıtı | `auto`, `npu`, `gpu` veya `cpu`. |
| `--json` | | Sonucun tamamını JSON olarak verir; dosya yolu `result_path` alanındadır. |

Sonuçların karşılaştırılabilir olması için her çalıştırma açgözlü kod çözme kullanır ve en fazla 256 yeni token üretir. Aşağıdaki, referans makinenin (Intel Core Ultra 7 268V, yonga 0.5.0) NPU'sundaki standart performans ölçümüdür:

```text
Model           Qwen/Qwen3-0.6B
Artifact        cc24636a2f49356a
Device          NPU
Profile         npu-int4-sym-g128
Suite           standard  (3 runs, 1 warmup)
Max new tokens  256
Compile         858 ms

Prompt               In tok   Out tok       TTFT   tok/s      TPOT    Wall
──────────────────────────────────────────────────────────────────────────
short-instruction        31       256   164.4 ms    68.2   14.7 ms   3.9 s
medium-explanation      104       256   164.7 ms    67.7   14.8 ms   3.9 s
long-context            623       256   165.0 ms    67.7   14.8 ms   3.9 s
Saved: …/benchmarks/20261011T170311Z-npu-standard-96ace441.json
```

Sütunlar şöyle okunur:

| Sütun | Anlamı |
|---|---|
| In tok / Out tok | Token cinsinden istem uzunluğu ve üretilen uzunluk. |
| TTFT | İlk token süresi (time to first token): istemin verilmesinden ilk üretilen token'a kadar geçen süre. İstemin tamamının okunmasını kapsar; yanıt başlamadan önce hissettiğiniz gecikme budur. |
| tok/s | Saniyede üretilen token. |
| TPOT | İlkinden sonraki her çıktı token'ı için geçen süre (time per output token); kabaca üretim hızının tersidir. |
| Wall | Bir çalıştırmanın toplam süresi. |
| Compile | İlk çalıştırmadan önce modelin derlenip yüklenme süresi. |

JSON dosyası `~/.local/share/yonga/benchmarks/` altına kaydedilir. Donanımı, paket sürümlerini, aygıt özelliklerini, üretim ayarlarını, her istemin özetini (hash) ve her bir çalıştırmayı kaydeder; böylece sonraki bir sonuç onunla adil biçimde karşılaştırılabilir. `bench` başarıda 0, bazı çalıştırmalar başarısız olduğunda 2, hiçbir çalıştırma ölçüm üretemediğinde 52 koduyla çıkar.

## eval ile nicemlemenin bedelini ölçmek {#eval}

`bench` bir artifact'in ne kadar hızlı olduğunu söyler; `eval` ise nicemlemenin modeli ne kadar değiştirdiğini. Aday modelin sonraki token tahminlerini, aynı ağırlıkların daha yüksek hassasiyetli bir **referans** artifact'inin tahminleriyle, aynı metni okuyarak karşılaştırır. Örnekleme yapılmadığından sonuç belirlenimcidir (deterministic).

Olağan akış, CPU'da neredeyse kayıpsız bir INT8 referans hazırlayıp INT4 artifact'inizi onunla karşılaştırmaktır:

```bash
yonga prepare Qwen/Qwen3-0.6B --device cpu --profile cpu-int8-asym-ref --yes
yonga list                                     # iki artifact kimliğini not edin
yonga eval INT4_ARTIFACT --reference REF_ARTIFACT
```

Üç şey raporlar; hem toplamda hem de yerleşik metin derleminin her parçası için (düzyazı, teknik yazı, Python, aritmetik, diyalog, Türkçe ve diğerleri):

| Ölçüt | Anlamı |
|---|---|
| Mean KL | Referanstan ortalama KL ıraksaması, nat cinsinden. Düşük olan daha yakındır; 0 aynı demektir. Artifact'leri sıralarken bakılacak sayı budur. |
| Top-1 agreement | İkisinin en olası sonraki token'da ne sıklıkla anlaştığı. |
| Perplexity | Her artifact'in derlem üzerindeki şaşkınlık (perplexity) değeri ve adayın referansa göre değişimi. |

Örneğin proje, Qwen3.5-9B'yi bir INT8 referansa karşı şöyle ölçtü ([PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#8-output-quality)):

| | `npu-int4-sym-cw` | `gpu-int4-asym-g128` |
|---|---|---|
| Mean KL (nat) | 0,1972 | 0,0799 |
| Top-1 agreement | %81,82 | %86,80 |
| Perplexity (referans 4,655) | 5,700 (+%22,44) | 4,983 (+%7,05) |

Seçenekler:

- `--device cpu` (varsayılan) tam hassasiyette çalışır; böylece ölçtüğünüz fark nicemlemeden gelir. `--device gpu` GPU'nun gerçekte ne hesapladığını ölçer. `--device npu` yoktur: bir NPU artifact'i de diğerleri gibi CPU'da değerlendirilir.
- `--corpus FILE` kendi metninizi okur (yerleşik biçimde bir `.json` ya da boş satırlardan bölünen düz `.txt`).
- `--allow-different-sources`, aynı modeli tutan farklı depolardan dönüştürülmüş artifact'leri karşılaştırır; örneğin bir GGUF sürümünü ve özgün ağırlıkları. yonga, kökene sizin kefil olduğunuzu kaydeder.
- `--json` sonucu JSON olarak verir.

Model 16 bit hassasiyette belleğe sığıyorsa referans olarak `cpu-fp16-ref` kullanın. Sonuç `~/.local/share/yonga/evaluations/` altına kaydedilir ve çıktı `Saved: PATH` ile biter. Daha fazlası [eval belirtiminde](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#13a-eval).

## Hugging Face Hub'a yayımlamak {#publish}

`publish`, hazırlanmış bir artifact'i, manifest'ten üretilen bir model kartıyla Hub'a yükler: revizyonlarıyla köken, dışa aktarma ayarları, onarımlar, hangi aygıtların yoklamayı geçtiği ve her dosyanın SHA-256 özeti. Yalnızca siz çağırdığınızda çalışır; başka hiçbir komut bir şey yüklemez.

```bash
yonga publish ARTIFACT --repo you/model-openvino-int4 --dry-run   # kart ve yükleme listesi, yükleme yok
yonga publish ARTIFACT --repo you/model-openvino-int4             # özel depo, önce sorar
```

- Ad alanına yazma yetkisi olan bir Hugging Face erişim belirteci (token) gerekir (`hf auth login` veya `HF_TOKEN`); yoksa 21 koduyla durur (`--dry-run` yalnızca uyarır).
- `--repo` her zaman zorunludur. `--public` vermediğiniz sürece depo özeldir; var olan bir depoya yalnızca `--update` ile yazılır.
- Lisanslar önce Hub'da denetlenir. Türev çalışmayı yasaklayan bir lisans reddedilir; yonga'nın sınıflandıramadığı bir lisans için, onu okuduğunuzu beyan eden `--license-reviewed` gerekir. `--yes` bunu asla kapsamaz.
- Yerel yollar, kullanıcı adınız ve makine adınız yüklenmez.

Her denetim için [publish belirtimine](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#14a-publish) bakın.

## Sonraki adımlar {#next-steps}

- Modeli başka araçlara sunun: [OpenAI uyumlu sunucu](serve.md).
- Bir şey başarısız olduysa: [Sorun giderme](../troubleshooting.md) ve [Çıkış kodları](../reference/exit-codes.md).
- Her komutun her seçeneği: [Komut başvurusu](../reference/cli.md).
