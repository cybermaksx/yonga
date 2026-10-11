# Komut başvurusu

yonga 0.5.1'deki her komut ve alt komut; seçenekleri ve varsayılanlarıyla, `yonga <komut> --help` çıktısında göründükleri gibi, her biri için bir örnekle birlikte. Hangi komutun ne zaman kullanılacağını kılavuzlar anlatır; bu sayfa bir şeye hızlıca bakmak içindir. Her komutun `--json` belgesi dahil tüm davranışı [CLI_SPEC.md](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md) belgesinde tanımlanmıştır (İngilizce).

Seçenek tablolarında **Varsayılan** sütunundaki kısa çizgi, `--help` çıktısının bir varsayılan göstermediği anlamına gelir: seçenek kapalıdır ya da açıklamada belirtildiği gibi yonga hazırlanmış modele (artifact), uyumluluk kayıt defterine (registry) veya yapılandırmaya göre karar verir. Her komut ve komut grubu ayrıca `--help` seçeneğini de kabul eder; aşağıda yinelenmemiştir.

## Genel seçenekler { #global-options }

```text
yonga [OPTIONS] COMMAND [ARGS]...
```

Intel AI PC'ler için yerel yapay zekâ modeli yöneticisi (OpenVINO, Optimum Intel, Hugging Face).

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--config` | `PATH` | `~/.config/yonga/config.toml` | Varsayılan yerine yüklenecek yapılandırma dosyası. |
| `--data-dir` | `PATH` | `~/.local/share/yonga` | Hazırlanmış artifact'lerin ve günlüklerin saklandığı yer. |
| `--cache-dir` | `PATH` | `~/.cache/yonga` | Yönetilen geçici dizinin ve derleme önbelleğinin bulunduğu yer. |
| `--offline` | | kapalı | Hugging Face Hub'a asla bağlanma. |
| `--json` | | kapalı | stdout'a süslemesiz tek bir JSON belgesi yaz. |
| `--verbose`, `-v` | | kapalı | Neler olduğunu açıkla. |
| `--debug` | | kapalı | Hatalara ham hata metnini de ekle. |
| `--version`, `-V` | | | Sürümü göster ve çık. |

Genel seçenekler komut adından **önce** gelir:

```bash
yonga --offline --debug prepare Qwen/Qwen3-0.6B
```

`--json`, komutların kendi adlarından sonra da kabul ettiği tek genel seçenektir (`yonga list --json`). `chat` ve `serve` onu 70 çıkış koduyla reddeder. Varsayılan dizinler, tanımlıysa `XDG_CONFIG_HOME`, `XDG_DATA_HOME` ve `XDG_CACHE_HOME` değişkenlerini izler; kullanılan dizinleri `yonga registry paths` yazdırır. Ayarlar önce komut satırından, sonra `YONGA_*` ortam değişkenlerinden, sonra yapılandırma dosyasından alınır.

Komutsuz çalıştırılan `yonga` komut listesini yazdırır ve 70 çıkış koduyla sonlanır; ayrıştırılamayan her komut satırı da (bilinmeyen bir seçenek, eksik bir argüman, yanlış türde bir değer) öyle. `--help` 0 ile sonlanır. Tüm kodlar için [Çıkış kodları](exit-codes.md) sayfasına bakın.

## Komutlara genel bakış { #commands-at-a-glance }

| Komut | Amaç |
|---|---|
| [`doctor`](#yonga-doctor) | Bu ana makineyi denetler: OpenVINO, aygıtlar, NPU erişimi, depolama ve kimlik doğrulama. |
| [`search`](#yonga-search) | Hugging Face Hub'da arar ve her sonuç hakkında kayıt defterinin bildiklerini gösterir. |
| [`inspect`](#yonga-inspect) | Bir modeli ve onu bu ana makinede hazırlamanın neler gerektireceğini anlatır. |
| [`prepare`](#yonga-prepare) | Bir modeli indirir, dönüştürür, doğrular, yoklar ve hazırlanmış bir artifact'e dönüştürür. |
| [`list`](#yonga-list) | Bu ana makinede hazırlanmış artifact'leri listeler. |
| [`remove`](#yonga-remove) | Hazırlanmış bir artifact'i ve derleme önbelleği girdilerini kaldırır. |
| [`probe`](#yonga-probe) | Hazırlanmış bir artifact'i bir aygıtta derler ve neyin çalıştığını bildirir. |
| [`chat`](#yonga-chat) | Hazırlanmış bir artifact ile yerel olarak sohbet eder. |
| [`serve`](#yonga-serve) | Hazırlanmış bir artifact'i OpenAI uyumlu bir HTTP API üzerinden sunar. |
| [`eval`](#yonga-eval) | CANDIDATE'in tahminlerinin REFERENCE'ınkilerden ne kadar uzaklaştığını ölçer (KL, top-1, perplexity). |
| [`publish`](#yonga-publish) | Hazırlanmış bir artifact'i oluşturulan bir model kartıyla Hugging Face Hub'a yükler. |
| [`bench`](#yonga-bench) | Hazırlanmış bir artifact'in performans ölçümünü yapar ve yeniden üretilebilir bir JSON sonucu kaydeder. |
| [`logs`](#yonga-logs) | Diskte tutulan iş günlüklerini en yeniden başlayarak gösterir. |
| [`registry`](#yonga-registry) | Uyumluluk kayıt defterini inceler, doğrular ve günceller. |
| [`env`](#yonga-env) | Dışa aktarmaların çalıştığı yönetilen ortamları listeler ve kaldırır. |
| [`cache`](#yonga-cache) | OpenVINO derleme önbelleğini gösterir ve budar. |
| [`self`](#yonga-self) | yonga AppImage'ını `PATH` üzerine 'yonga' olarak koyar ya da yeniden kaldırır. |

Bir komut `MODEL_OR_ARTIFACT` ya da `ARTIFACT` aldığında `yonga list` çıktısındaki bir artifact kimliğini (ya da en az altı karakterlik benzersiz bir önekini), artifact'in dizinini veya bir Hugging Face depo kimliğini verebilirsiniz. Birden çok hazırlanmış artifact'i olan bir depo kimliğinde, yalnızca okuyan komutlar en yeni hazır olanı kullanır; `remove` ise reddeder ve hepsini listeler.

## yonga doctor

```text
yonga doctor [OPTIONS]
```

Bu ana makineyi denetler: OpenVINO, aygıtlar, NPU erişimi, depolama ve kimlik doğrulama.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--full` | | kapalı | Yüklenen kayıt defterini de bildir. |
| `--json` | | kapalı | Raporu JSON olarak ver. |

Çalışma zamanı ve aygıt kullanılabiliyor ve ilgilenilmesi gereken bir şey yoksa 0, uyarılarla birlikte kullanılabiliyorsa 2, kullanılabilir bir OpenVINO çalışma zamanı ya da aygıt yoksa 10 çıkış koduyla sonlanır. Her bölümü [Bakım](../guide/maintain.md#check-the-host-with-doctor) sayfası açıklar.

```bash
yonga doctor --full
```

## yonga search

```text
yonga search [OPTIONS] QUERY
```

Hugging Face Hub'da arar ve her sonuç hakkında kayıt defterinin bildiklerini gösterir.

| Argüman | Açıklama |
|---|---|
| `QUERY` | Hub'da aranacak sözcükler. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--limit` | tamsayı, 1–100 | `20` | Gösterilecek sonuç sayısı. |
| `--device` | metin | – | Tek bir aygıt için uyumluluğu göster: `npu`, `gpu`, `cpu` ya da `auto`. |
| `--json` | | kapalı | Sonuçları JSON olarak ver. |

```bash
yonga search qwen3 --device npu --limit 10
```

## yonga inspect

```text
yonga inspect [OPTIONS] MODEL
```

Bir modeli ve onu bu ana makinede hazırlamanın neler gerektireceğini anlatır. Hiçbir ağırlık dosyası indirmez.

| Argüman | Açıklama |
|---|---|
| `MODEL` | Hugging Face depo kimliği. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--revision` | metin | – | Dal, etiket ya da commit sha. |
| `--files` | | kapalı | Depodaki dosyaları da listele. |
| `--json` | | kapalı | Raporu JSON olarak ver. |

```bash
yonga inspect Qwen/Qwen3-0.6B
```

## yonga prepare

```text
yonga prepare [OPTIONS] MODEL
```

Bir modeli indirir, dönüştürür, doğrular, yoklar ve hazırlanmış bir artifact'e dönüştürür. [Model hazırlama](../guide/prepare.md) sayfasına bakın.

| Argüman | Açıklama |
|---|---|
| `MODEL` | Hugging Face depo kimliği. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--revision` | metin | – | Dal, etiket ya da commit sha. |
| `--device` | metin | `auto` | `auto`, `npu`, `gpu` ya da `cpu`. |
| `--profile` | metin | – | Dönüştürme profili kimliği (`yonga registry profiles` listeler). Verilmezse kayıt defteri seçer. |
| `--source` | metin | – | İstenen depo yerine bu depoyu dönüştür. |
| `--trust-remote-code` | | kapalı | Dönüştürmenin depoda yayımlanan Python kodunu (uzak kod) çalıştırmasına izin ver. |
| `--ignore-exporter-requirements` | | kapalı | Bildirilen bir dışa aktarıcı paket sürümü karşılanmasa da ya da kayıt defteri bu dışa aktarmanın kurulu sürümlerle başarısız olduğunu bilse de dönüştür. Bilinçli ve ayrı bir seçimdir: `--yes` onu asla açmaz. |
| `--no-managed-env` | | kapalı | Asla yönetilen bir dışa aktarma ortamı kurma; kurulu paketler dışa aktarıcı gereksinimlerini karşılayamıyorsa reddet. |
| `--gguf-file` | metin | – | Bir GGUF deposu için içe aktarılacak dosya. Varsayılan olarak yonga, tensör türlerini OpenVINO GenAI'ın okuyabildiği ilk tercih edilen nicemleme (quantization) dosyasını seçer. |
| `--fresh-export` | | kapalı | Başarısız bir iş, aynı girdilerle tamamlanmış ve yeniden kullanılabilecek bir dışa aktarma bırakmış olsa bile yeniden dışa aktar. |
| `--yes`, `-y` | | kapalı | Olağan sorulara evet yanıtı ver. |
| `--force` | | kapalı | Bir artifact zaten olsa bile yeniden hazırla. |
| `--no-repair` | | kapalı | Dışa aktarılan IR'yi asla değiştirme. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu
```

## yonga list

```text
yonga list [OPTIONS]
```

Bu ana makinede hazırlanmış artifact'leri listeler.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--clean` | | kapalı | Eskimiş hazırlık ve geçici dizinleri saklama ilkesine göre kaldır. |
| `--json` | | kapalı | Kataloğu JSON olarak ver. |

```bash
yonga list
```

## yonga remove

```text
yonga remove [OPTIONS] ARTIFACT
```

Hazırlanmış bir artifact'i ve derleme önbelleği girdilerini kaldırır. Hugging Face indirme önbelleğine asla dokunulmaz.

| Argüman | Açıklama |
|---|---|
| `ARTIFACT` | Artifact kimliği, depo kimliği ya da yol. Tam olarak bir artifact ile eşleşmelidir. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--yes`, `-y` | | kapalı | Onay isteme. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga remove cc24636a2f49356a
```

## yonga probe

```text
yonga probe [OPTIONS] MODEL_OR_ARTIFACT
```

Hazırlanmış bir artifact'i bir aygıtta derler ve neyin çalıştığını bildirir (yoklama).

| Argüman | Açıklama |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact kimliği ya da yolu. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--device` | metin | `auto` | `auto`, `npu`, `gpu` ya da `cpu`. |
| `--all-submodels` | | kapalı | Yalnızca dil modelini değil, her alt modeli derle. |
| `--quick` | | kapalı | GenAI işlem hattını asla kurma; ertelenmiş bir ham derleme sonuçsuz (inconclusive) olarak bildirilir. |
| `--timeout` | `SECONDS` | – | Ertelenmiş bir derlemeyi sonuca bağlayan GenAI işlem hattı kurulumunu sınırla: bir alt süreçte çalıştır ve SECONDS sonra durdur (aygıt o zaman sonuçsuz sayılır). `--quick` ile birlikte kullanılamaz. |
| `--json` | | kapalı | Raporu JSON olarak ver. |

Herhangi bir aygıt geçtiyse 0; hiçbiri geçmediği hâlde hiçbiri de açıkça başarısız olmadıysa (`--quick` ya da `--timeout` nedeniyle sonuçsuz kaldıysa) 2; aksi hâlde ilk başarısızlığın koduyla sonlanır.

```bash
yonga probe cc24636a2f49356a --device npu
```

## yonga chat

```text
yonga chat [OPTIONS] MODEL_OR_ARTIFACT
```

Hazırlanmış bir artifact ile yerel olarak sohbet eder. `/reset` ve `/exit` gibi sohbet içi komutlar için [Sohbet, yoklama, ölçüm ve değerlendirme](../guide/run.md) sayfasına bakın.

| Argüman | Açıklama |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact kimliği ya da yolu. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--device` | metin | – | `auto`, `npu`, `gpu` ya da `cpu`. Verilmezse bir aygıt belirtiyorsa `default_device` ayarı, aksi hâlde artifact'in önerilen aygıtı. |
| `--max-new-tokens` | tamsayı | – | Çıktı token sınırı. |
| `--temperature` | sayı | – | Örnekleme sıcaklığı. |
| `--top-p` | sayı | – | Çekirdek (nucleus) örnekleme olasılığı. |
| `--top-k` | tamsayı | – | Top-k örnekleme kesme değeri. |
| `--show-reasoning` | | kapalı | Gizli akıl yürütme bloklarını göster. |
| `--system` | metin | – | Oturumun sistem istemi. |
| `--json` | | | Desteklenmez: bir konuşma tek bir belge değildir (çıkış kodu 70). |

```bash
yonga chat cc24636a2f49356a
```

## yonga serve

```text
yonga serve [OPTIONS] MODEL_OR_ARTIFACT
```

Hazırlanmış bir artifact'i OpenAI uyumlu bir HTTP API üzerinden sunar. pip kurulumunda `serve` ek paket grubunu gerektirir; AppImage bunu içerir. [OpenAI uyumlu sunucu](../guide/serve.md) sayfasına bakın.

| Argüman | Açıklama |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact kimliği ya da yolu. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--host` | metin | `127.0.0.1` | Sunucunun bağlanacağı (bind) adres. Siz aksini söylemedikçe yerel döngü (loopback) adresi. |
| `--port` | tamsayı, 1–65535 | `8000` | Dinlenecek TCP bağlantı noktası. |
| `--device` | metin | – | `auto`, `npu`, `gpu` ya da `cpu`. Verilmezse bir aygıt belirtiyorsa `default_device` ayarı, aksi hâlde artifact'in önerilen aygıtı. |
| `--max-new-tokens` | tamsayı | – | Bir istek belirtmediğinde varsayılan çıktı token sınırı. |
| `--show-reasoning` | | kapalı | Akıl yürütme bloklarını gizlemek yerine döndür. |
| `--max-body-mb` | tamsayı, 1–64 | `8` | MiB cinsinden en büyük istek gövdesi; gömülü görseller de buna dahildir. |
| `--carry-images-forward` | | kapalı | Önceki mesajlarla gönderilen görselleri reddetmek yerine son mesaja taşı. |
| `--json` | | | Desteklenmez: bir sunucu tek bir belge değildir (çıkış kodu 70). |

```bash
yonga serve cc24636a2f49356a --port 8000
```

## yonga eval

```text
yonga eval [OPTIONS] CANDIDATE
```

CANDIDATE'in tahminlerinin REFERENCE'ınkilerden ne kadar uzaklaştığını ölçer (KL, top-1, perplexity). İkisi de aynı ağırlıklardan hazırlanmış artifact'ler olmalıdır; referans genellikle daha yüksek duyarlıklı bir artifact'tir, örneğin `--profile cpu-int8-asym-ref` ile hazırlanmış olan.

| Argüman | Açıklama |
|---|---|
| `CANDIDATE` | Değerlendirilecek artifact. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--reference` | metin | – | Karşılaştırılacak, aynı ağırlıklardan daha yüksek duyarlıklı artifact. Zorunlu. |
| `--device` | metin | `cpu` | `cpu` (tam duyarlık) ya da `gpu`. |
| `--corpus` | yol | – | Birlikte gelen derlem yerine okunacak metin (`.json` ya da `.txt`). |
| `--allow-different-sources` | | kapalı | Farklı depolardan dönüştürülmüş artifact'leri karşılaştır (ör. GGUF ile özgün). |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga eval CANDIDATE_ID --reference REFERENCE_ID
```

## yonga publish

```text
yonga publish [OPTIONS] ARTIFACT
```

Hazırlanmış bir artifact'i oluşturulan bir model kartıyla Hugging Face Hub'a yükler. `--public` vermediğiniz sürece depo özeldir.

| Argüman | Açıklama |
|---|---|
| `ARTIFACT` | Hazırlanmış artifact kimliği ya da yolu. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--repo` | metin | – | Hub'daki hedef `NAMESPACE/NAME`. Her zaman açıkça verilir. Zorunlu. |
| `--public` | | kapalı | Herkese açık bir depo oluştur (varsayılan özeldir). |
| `--update` | | kapalı | Zaten var olan bir depoya yaz. |
| `--license-reviewed` | | kapalı | yonga'nın sınıflandıramadığı bir lisansı okuduğunuzu ve lisansın nicemlenmiş bir türevin yeniden dağıtımına izin verdiğini beyan et. `--yes` bunu asla kapsamaz. |
| `--dry-run` | | kapalı | Her şeyi denetle ve kartı yerel olarak yaz; hiçbir şey yükleme. |
| `--output` | yol | – | Kartın, manifestin ve yükleme listesinin yazılacağı yer. `--dry-run` verilip `--output` verilmezse `./yonga-publish-<artifact>`. |
| `--yes`, `-y` | | kapalı | Yüklemeyi soru sormadan onayla. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga publish cc24636a2f49356a --repo NAMESPACE/Qwen3-0.6B-openvino-npu --dry-run
```

## yonga bench

```text
yonga bench [OPTIONS] MODEL_OR_ARTIFACT
```

Hazırlanmış bir artifact'in performans ölçümünü yapar ve yeniden üretilebilir bir JSON sonucu kaydeder.

| Argüman | Açıklama |
|---|---|
| `MODEL_OR_ARTIFACT` | Artifact kimliği ya da yolu. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--device` | metin | – | `auto`, `npu`, `gpu` ya da `cpu`. Verilmezse artifact'in önerilen aygıtı. |
| `--suite` | metin | `standard` | `quick`, `standard` ya da `long`. |
| `--runs` | tamsayı | – | İstem başına ölçülen çalıştırma sayısı. Verilmezse `bench_runs` ayarı (3). |
| `--warmups` | tamsayı | – | İstem başına ısınma çalıştırması sayısı. Verilmezse `bench_warmups` ayarı (1). |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga bench cc24636a2f49356a --suite quick
```

## yonga logs

```text
yonga logs [OPTIONS] [JOB_ID]
```

Diskte tutulan iş günlüklerini en yeniden başlayarak gösterir. `JOB_ID` verilmezse en son 20 işi listeler.

| Argüman | Açıklama |
|---|---|
| `JOB_ID` | İş kimliği ya da bir öneki. İsteğe bağlı. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--raw` | | kapalı | JSON Lines dosyasını olduğu gibi yazdır. |
| `--bundle` | | kapalı | Hata bildirimi için hassas bilgileri ayıklanmış, anonimleştirilmiş bir `.tar.gz` yaz (günlük, durum, manifest, ana makine raporu). |
| `--output` | dosya | `./yonga-report-<job>.tar.gz` | `--bundle` arşivinin yazılacağı yer. |
| `--json` | | kapalı | Kayıtları JSON olarak ver. |

```bash
yonga logs de862650 --bundle
```

## yonga registry

```text
yonga registry [OPTIONS] COMMAND [ARGS]...
```

Uyumluluk kayıt defterini inceler, doğrular ve günceller. [Bakım](../guide/maintain.md#the-compatibility-registry) sayfasına bakın.

### registry validate

```text
yonga registry validate [OPTIONS]
```

Kayıt defterinin her katmanını yükler ve kendi içinde tutarlı olduğunu denetler. Geçerliyse 0, kurulu bir uzak paket yok sayıldıysa 2, sorun bulunursa 70 çıkış koduyla sonlanır.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga registry validate
```

### registry profiles

```text
yonga registry profiles [OPTIONS]
```

`--profile` seçeneğinin kabul ettiği dönüştürme profillerini listeler.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--device` | metin | – | Yalnızca bu aygıtın profilleri. |
| `--json` | | kapalı | Profilleri JSON olarak ver. |

```bash
yonga registry profiles --device npu
```

### registry explain

```text
yonga registry explain [OPTIONS] MODEL
```

Bir modelle hangi kayıt defteri kurallarının eşleştiğini ve neden kazandıklarını gösterir.

| Argüman | Açıklama |
|---|---|
| `MODEL` | Hugging Face depo kimliği. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--device` | metin | – | Yalnızca bu aygıt için açıkla. |
| `--json` | | kapalı | Açıklamayı JSON olarak ver. |

```bash
yonga registry explain Qwen/Qwen3-0.6B --device npu
```

### registry paths

```text
yonga registry paths [OPTIONS]
```

Kayıt defteri katmanlarının, yapılandırmanın ve yönetilen verilerin nerede durduğunu gösterir.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--json` | | kapalı | Yolları JSON olarak ver. |

```bash
yonga registry paths
```

### registry update

```text
yonga registry update [OPTIONS]
```

İmzalı bir uzak kayıt defteri paketini indirir, doğrular ve kurar. Yalnızca `~/.config/yonga/registry-keys/` dizinindeki bir anahtarla imzalanmış paketler kabul edilir.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--url` | metin | `registry_url` ayarı (tanımsız) | Paket URL'si (https); imza URL.sig adresinden indirilir. |
| `--file` | yol | – | Diskte duran bir paket. `--url` ile birlikte kullanılamaz. |
| `--signature` | yol | `FILE.sig` | Paketin imzası. Yalnızca `--file` ile. |
| `--yes`, `-y` | | kapalı | Soru sormadan kur. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga registry update --file bundle.json
```

### registry status

```text
yonga registry status [OPTIONS]
```

Kurulu uzak kayıt defteri paketini ve hâlâ doğrulanıp doğrulanmadığını gösterir. Kurulu bir paket yok sayılıyorsa 2 çıkış koduyla sonlanır.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--json` | | kapalı | Durumu JSON olarak ver. |

```bash
yonga registry status
```

### registry reset

```text
yonga registry reset [OPTIONS]
```

Kurulu uzak paketi kaldırır ve yerleşik kayıt defterine geri döner.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga registry reset
```

## yonga env

```text
yonga env [OPTIONS] COMMAND [ARGS]...
```

Dışa aktarmaların çalıştığı yönetilen ortamları listeler ve kaldırır. [Bakım](../guide/maintain.md#managed-export-environments) sayfasına bakın.

### env list

```text
yonga env list [OPTIONS]
```

Yönetilen dışa aktarma ortamlarını ve kapladıkları disk alanını gösterir.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--json` | | kapalı | Ortamları JSON olarak ver. |

```bash
yonga env list
```

### env remove

```text
yonga env remove [OPTIONS] KEY
```

Yönetilen bir dışa aktarma ortamını siler. Sonraki bir dışa aktarma gerektirirse yeniden kurulur.

| Argüman | Açıklama |
|---|---|
| `KEY` | Ortam anahtarı ya da benzersiz bir öneki. Zorunlu. |

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--yes`, `-y` | | kapalı | Onay isteme. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga env remove fbfac3ac91ba
```

## yonga cache

```text
yonga cache [OPTIONS] COMMAND [ARGS]...
```

OpenVINO derleme önbelleğini gösterir ve budar. Alt komutsuz `yonga cache`, `yonga cache list` ile aynıdır. [Bakım](../guide/maintain.md#the-compile-cache) sayfasına bakın.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--json` | | kapalı | Girdileri JSON olarak ver. |

### cache list

```text
yonga cache list [OPTIONS]
```

Derleme önbelleği girdilerini listeler: boyut, son kullanım, sahip ve hâlâ kullanılıp kullanılmadıkları.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--json` | | kapalı | Girdileri JSON olarak ver. |

```bash
yonga cache list
```

### cache prune

```text
yonga cache prune [OPTIONS]
```

Sahipsiz (`orphaned`) ve eski çalışma zamanına ait (`stale-runtime`) derleme önbelleği girdilerini kaldırır (istenirse daha fazlasını da).

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--older-than` | `DAYS` | – | Bu kadar gündür kullanılmayan her girdiyi de kaldır. |
| `--unattributed` | | kapalı | Hiçbir artifact'e bağlanamayan girdileri de kaldır. |
| `--all` | | kapalı | Derleme önbelleğinin tamamını boşalt. |
| `--dry-run` | | kapalı | Neyin gideceğini göster; hiçbir şey silme. |
| `--yes`, `-y` | | kapalı | Onay isteme. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga cache prune --dry-run
```

## yonga self

```text
yonga self [OPTIONS] COMMAND [ARGS]...
```

yonga AppImage'ını `PATH` üzerine 'yonga' olarak koyar ya da yeniden kaldırır. Yalnızca yonga AppImage'dan çalışırken işe yarar.

### self install

```text
yonga self install [OPTIONS]
```

Çalışan AppImage'ı `~/.local/bin/yonga` konumuna kopyalar; böylece 'yonga' her kabukta çalışır.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--bin-dir` | `DIR` | `~/.local/bin` | 'yonga' komutunun kurulacağı dizin. |
| `--yes`, `-y` | | kapalı | Eski bir AppImage'ı sormadan değiştir. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
./yonga-0.5.1-x86_64.AppImage self install
```

### self uninstall

```text
yonga self uninstall [OPTIONS]
```

`~/.local/bin/yonga` konumundaki AppImage kopyasını kaldırır. Modeller ve önbellekler korunur.

| Seçenek | Değer | Varsayılan | Açıklama |
|---|---|---|---|
| `--bin-dir` | `DIR` | `~/.local/bin` | 'yonga' komutunun kaldırılacağı dizin. |
| `--yes`, `-y` | | kapalı | Onay isteme. |
| `--json` | | kapalı | Sonucu JSON olarak ver. |

```bash
yonga self uninstall
```
