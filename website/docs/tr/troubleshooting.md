# Sorun giderme

Bir şeyler ters gittiğinde yonga neyin, neden başarısız olduğunu ve sırada ne çalıştırmanız
gerektiğini söyler, ham hata metnini de bir iş günlüğünde saklar. Bu sayfa bu çıktının nasıl
okunacağını gösterir ve kullanıcıların gerçekten karşılaştığı sorunları, göreceğiniz tanı koduyla
ve çözümüyle birlikte tek tek ele alır. Yerleşik kayıt defterinin üretebileceği her kodu içeren tam
başvuru, GitHub'daki
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md) belgesidir.

## `yonga doctor` ile başlayın { #doctor }

Sorun ne olursa olsun, önce şunu çalıştırın:

```bash
yonga doctor
```

`doctor` ana makineyi, OpenVINO paketlerini, aygıtları, NPU erişimini, depolamayı ve Hugging Face
kimlik doğrulamasını denetler; hiçbir şeyi değiştirmez. Her satır `✓` (sorun yok), `!` (uyarı),
`✗` (hata) ya da `·` (bilgi) ile işaretlenir ve bir sorunun altında ne yapmanız gerektiğini
söyleyen `→` satırları bulunur. Sağlıklı bir makinenin NPU bölümü, kısaltılmış olarak şöyle görünür:

```text
NPU
✓  /dev/accel/accel0   accessible  group render
✓  Group               render  the current user is a member
✓  driver              1789049548
✓  compiler            524291
✓  compiler-in-plugin  available  the NPU plugin compiles graphs itself
…
```

Çıkış kodu durumu özetler: **0** her şey kullanılabilir, **2** uyarılarla kullanılabilir, **10**
yonga'nın ihtiyaç duyduğu bir şey eksik. `yonga doctor --full` ayrıca yüklü kayıt defterini
raporlar ve daha önce başarısız olduysa dışa aktarıcının içe aktarılmasını yeniden denetler.

## Bir tanıyı okumak { #diagnostic }

Bir arıza, tanı olarak yazdırılır. Bu örnek, yerel önbellekte bulunmayan bir model için `--offline`
ile çalıştırılan `inspect` komutundan geliyor:

```text
✗ Not available offline

Offline mode is active and Qwen/Qwen3-0.6B is not in the local Hugging Face cache
(~/.cache/huggingface/hub), so it cannot be used without network access.

  - Re-run without offline mode (no --offline, and YONGA_OFFLINE / HF_HUB_OFFLINE unset) to
populate the cache: …
  - Or point HF_HOME at a cache that already contains this revision.
```

Yukarıdan aşağıya okuyun:

- **Başlık** neyin başarısız olduğunu söyler, **ileti** de bunu sade bir dille açıklar.
- **`-` ile başlayan adımlar** sırasıyla yapılacaklardır. Bir komut adlandıran adım size gerçek bir
  komut verir: önerdiği değişikliği içeren, kopyalanmaya hazır kendi komutunuz.
- **`Recommended fallback`** ve **`Run:`**, sorun tek bir aygıtla ilgili olduğunda görünür ve
  bunun yerine kullanılacak aygıtı ve komutu adlandırır.
- **`Job log:`**, iş çalıştıran komutlarda (`prepare`, `probe`, `bench`) günlüğün yoludur.
- **`Re-run with --debug to see the raw error`**, tanının arkasında bir derleyici iletisi gibi ham
  araç çıktısı olduğunda görünür.

`hub.offline.missing` gibi kararlı **tanı kodu**, `--json` çıktısında ve iş günlüğünde yer alır.
[Çıkış kodları](reference/exit-codes.md) sayfası kodları çıkış kodlarıyla eşleştirir.

Günlüklere bakmak için:

```bash
yonga logs                 # tüm işler, en yenisi önce, nasıl bittikleriyle
yonga logs JOB_ID          # tek bir iş, tablo olarak; kimliğin benzersiz her öneki işe yarar
yonga logs JOB_ID --raw    # JSON Lines dosyası, yazıldığı gibi
```

Ham hatayı terminalde görmek için komutu `--debug` ile yeniden çalıştırın. Bu genel (global) bir
seçenektir, bu yüzden komut adından **önce** gelir:

```bash
yonga --debug prepare Qwen/Qwen3-0.6B --device npu
```

## Sık karşılaşılan sorunlar { #common }

### NPU yok ya da erişilemiyor { #npu-access }

**Ne görürsünüz.** `doctor`, `/dev/accel/accel0` için eksik ya da `not accessible` der.
OpenVINO hiç NPU göstermiyorsa `prepare --device npu`, `device.not_present` ile (çıkış kodu 50)
durur; aygıtı açamayan bir komut ise `npu.device.permission_denied` ile (çıkış kodu 10) başarısız
olur.

**Neden.** Üç koşulun sağlanması gerekir: `intel_vpu` çekirdek modülü yüklü olmalı,
`/dev/accel/accel0` aygıt düğümü bulunmalı ve kullanıcınızın onu açma izni olmalı. Düğüm `render`
grubuna aittir ve sıradan bir hesap varsayılan olarak bu grupta değildir. Kullanıcı alanındaki NPU
sürücüsü, Intel'in Level Zero NPU sürücüsüdür ve bir dağıtım paketidir (Ubuntu referans makinesinde
`intel-level-zero-npu`); `pip install openvino` onu kurmaz.

**Çözüm.** İzin almak için kendinizi gruba ekleyin, sonra oturumu kapatıp yeniden açın:

```bash
sudo usermod -aG render $USER
```

yonga hiçbir zaman `sudo` çalıştırmaz ve izinleri kendisi değiştirmez; yalnızca komutu yazdırır.
Bir izin hatasında işi başka bir aygıta da taşımaz, çünkü bu, düzeltilmesini isteyeceğiniz bir ana
makine sorununu gizlerdi. NPU kurulumunu adım adım [Kurulum](installation.md) sayfası anlatır.

### Hızlandırıcı sürücüsü başlatılamıyor { #driver }

**Ne görürsünüz.** `device.driver_missing` (çıkış kodu 10), çoğunlukla
`ZE_RESULT_ERROR_UNINITIALIZED` gibi bir hatadan.

**Çözüm.** `yonga doctor` çalıştırın: sürücüyü, derleyiciyi ve çekirdek modülünün durumunu
gösterir. İzin sorunu gibi bu da ana makinenin kendisinin bozuk olduğunu gösterir; bu yüzden `prepare` başka
bir aygıta geri dönüş (fallback) yapmak yerine durur.

!!! note "Not"
    Bir NPU derleme hatasındaki `ZE_RESULT_ERROR_INVALID_ARGUMENT` bir sürücü sorunu **değildir**:
    sürücü yanıt vermiş ve derleyici çalışmıştır. [NPU modeli reddediyor](#npu-compiler) bölümüne
    bakın.

### `doctor` küçük bir `/tmp` için uyarıyor { #tmp }

**Ne görürsünüz.** `doctor`, Storage altında bir uyarıyla 2 ile çıkar:

```text
!  system temp          15.4 GiB tmpfs  7.0 GiB free at /tmp
                        → yonga runs conversions in a managed, disk-backed
                        temporary directory instead of the system temp
                        filesystem, so a small tmpfs is a warning and never
                        fatal.
…
```

**Çözüm.** Gerek yok; tmpfs'i yonga için büyütmeyin. Ubuntu'da `/tmp` RAM üzerinde durur, bu yüzden
yonga dönüştürmeyi önbellek altındaki kendi dizininde (`~/.cache/yonga/tmp/<job-id>`) yapar. Önemli
olan, `~/.cache/yonga` dizinini barındıran diskteki boş alandır.

### Disk alanı yetmiyor { #disk }

**Ne görürsünüz.** `prepare`, hiçbir şey indirmeden önce gereken alanı denetler ve
`preflight.insufficient_disk` ile (çıkış kodu 30) reddeder:

```text
✗ Not enough temporary space

The managed temporary directory <cache>/tmp/<job-id> has 15.1 GiB free, but this conversion needs
25.0 GiB (20.0 GiB plus a 5.0 GiB safety margin).

  - Free space on the cache filesystem.
  - Or point YONGA_CACHE_DIR at a larger filesystem.
```

Disk yine de bir dönüştürme sırasında dolarsa `storage.insufficient_disk` ya da
`disk.insufficient_space` (çıkış kodu 30) alırsınız. OpenVINO dolu bir diski çoğu zaman
`basic_ios::clear: iostream error` olarak bildirir; bu bir dosya biçimi hatasına benzer ama değildir.

**Çözüm.** Yer açın ya da yonga'nın dizinlerini daha büyük bir diske taşıyın: tek bir komut için
genel `--cache-dir` ve `--data-dir` seçenekleriyle, tüm komutlar için şununla:

```bash
export YONGA_CACHE_DIR=/mnt/big/yonga-cache    # geçici dosyalar, derleme önbelleği, dışa aktarma ortamları
export YONGA_DATA_DIR=/mnt/big/yonga-data      # hazırlanmış modeller, günlükler, performans ölçümleri
```

Diski dolduran çoğu zaman derleme önbelleği ve eski dışa aktarma ortamlarıdır;
[derleme önbelleği büyüyüp duruyor](#compile-cache) bölümüne ve [Bakım](guide/maintain.md)
sayfasına bakın.

### Erişimi kısıtlı (gated) ya da özel bir model (401) { #gated }

**Ne görürsünüz.** `inspect` çalışır ama `prepare`, `hub.auth.gated` ya da `hub.auth.required`
ile (çıkış kodu 21) reddeder. Hiç erişim belirteci (token) yoksa `prepare` daha planda, indirmeden
önce reddeder.

**Çözüm.** Modelin Hugging Face Hub sayfasını açın, koşullarını kabul edin ve yonga'ya bir token
verin: ya `hf auth login` ile oturum açın ya da token'ı dışa aktarın:

```bash
export HF_TOKEN=hf_...
```

`doctor`'ın Hugging Face bölümü, yonga'nın bir kimlik bilgisi görüp görmediğini gösterir. Hub'ın,
var olmayan bir depoya yapılan anonim isteğe 404 değil 401 ile yanıt verdiğini unutmayın; yanlış
yazılmış bir ad erişim sorunu gibi görünebilir: `hub.repo.not_found` (çıkış kodu 20) iki durumu da
kapsar.

### Çevrimdışısınız ve model önbellekte yok { #offline }

**Ne görürsünüz.** `--offline` ile (ya da `YONGA_OFFLINE` veya `HF_HUB_OFFLINE` tanımlıyken)
`inspect` ya da `prepare`, [Bir tanıyı okumak](#diagnostic) bölümündeki örnekteki gibi
`hub.offline.missing` ile (çıkış kodu 20) başarısız olur.

**Çözüm.** Çevrimdışı kip Hub'a hiç bağlanmaz, bu yüzden model yerel Hugging Face önbelleğinde
bulunmalıdır. Modeli indirmek için komutu bir kez `--offline` olmadan çalıştırın ya da `HF_HOME`'u
modeli içeren bir önbelleğe yöneltin. Bir modelin ilk dönüştürmesi ayrıca ağdan kurulan bir
yönetilen dışa aktarma ortamı gerektirebilir; çevrimdışıyken yeni bir ortam `export.env.offline`
ile (çıkış kodu 20) başarısız olur, var olan bir ortam ise yeniden kullanılır.

### NPU modeli reddediyor { #npu-compiler }

**Ne görürsünüz.** `npu.compiler.` ya da `npu.pipeline.` ile başlayan bir kod, örneğin bir Qwen3.5
modeli için `npu.compiler.unsupported_graph.qwen35`. Ham metni şöyle görünür:

```text
[ERROR] [vpux-compiler] OneShotBufferizeVPU2VPUIP failed : Only supports FP16.
```

`probe`, `chat` ve `bench` 51 ile çıkar; `prepare --device npu` 50 ile çıkar;
`prepare --device auto` ise arızayı açıklar ve GPU'ya ya da CPU'ya geçmeden önce sorar.

**Neden.** NPU derleyicisi üç aygıt arasında en katı olanıdır ve bazı graflar onda derlenmez. Bu, modelin
bu derleyicideki bir sınırıdır, bozuk bir sürücü değildir: `doctor` NPU'yu yine sağlıklı gösterir
ve sürücüyü yeniden kurmak işe yaramaz.

**Çözüm.** GPU'yu (ya da CPU'yu) kullanın. Tanı tam komutu yazdırır; bir model için şudur:

```bash
yonga prepare MODEL --device gpu
```

Bu modelin GPU'da yoklamayı geçmiş bir hazırlanmış modeli (artifact) zaten varsa, onu orada
çalıştırın, örneğin `yonga chat ARTIFACT_ID --device gpu`.

!!! tip "İpucu"
    NPU'daki bir yoklama tüm işlem hattını (pipeline) kurar; büyük bir modelde bu dakikalar
    sürebilir. `yonga probe ARTIFACT_ID --device npu --quick` kurulumu atlar, `--timeout 600` ise
    600 saniye sonra durdurur. İkisi de NPU'yu `inconclusive` (sonuçsuz, çıkış kodu 2) bırakabilir;
    bu bir hüküm değildir. Kesin sonuç için yoklamayı `--quick` olmadan ya da daha uzun bir
    `--timeout` ile yeniden çalıştırın.

### Profilin grup boyutu modele uymuyor { #group-size }

**Ne görürsünüz.** `export.quantization.group_size_mismatch` (çıkış kodu 70). `prepare` bunu
genellikle modelin `config.json` dosyasından önceden kestirir ve indirmeden önce, planda,
`No GPU profile can quantize this model` gibi bir başlıkla durur.

**Neden.** Grup bazlı INT4 nicemleme (quantization), her ağırlık satırını 128'lik gruplara böler;
bu yüzden 128'in modelin katman genişliklerini tam bölmesi gerekir. Gizli boyutu 960 olan
`HuggingFaceTB/SmolLM2-360M-Instruct` gibi bazı modeller buna uymaz. Modelde ya da aygıtta bir
sorun yoktur.

**Çözüm.** İletinin adlandırdığı aygıtı ya da profili kullanın. Grup boyutu olmayan profiller
CPU'da `cpu-int8-asym-ref` ile `cpu-fp16-ref`, NPU'da ise `npu-int4-sym-cw`'dir:

```bash
yonga prepare MODEL --device cpu --profile cpu-int8-asym-ref
yonga prepare MODEL --device npu --profile npu-int4-sym-cw
```

`yonga registry profiles` her profili aygıtı ve ağırlık biçimiyle birlikte listeler.

### Python 3.14'te phi3 ya da mistral modelleri { #python314 }

**Ne görürsünüz.** `phi3`, `mistral`, `mixtral`, `gpt2` ve `bloom` gibi model türleri için,
indirmeden önce `export.optimum.normalized_config` (çıkış kodu 10).

**Neden.** optimum 2.3.0 ve öncesi, bu mimarilerin dışa aktarma yapılandırmasını Python 3.14'te
kuramaz. Hata optimum'dadır; optimum bunu henüz hiçbir sürümde yer almayan bir çekme isteğiyle
(pull request) düzeltmiştir. AppImage Python 3.14 içerdiği için o da bundan etkilenir.

**Çözüm.** yonga'yı Python 3.11, 3.12 ya da 3.13'ten, bir sanal ortama pip ile kurarak çalıştırın:

```bash
python3.13 -m venv ~/.venvs/yonga
~/.venvs/yonga/bin/pip install "yonga[openvino,export]"
```

Ya da başka bir mimariden bir model seçin; `yonga search QUERY` kayıt defterinin neyi bildiğini
gösterir.

### torch ile torchvision uyuşmuyor (pip kurulumları) { #torchvision }

**Ne görürsünüz.** Ön denetimde, indirmeden önce `export.torchvision.build_mismatch` (çıkış kodu
10) ve `doctor`'da bir `exporter import` uyarısı. Ham hata
`operator torchvision::nms does not exist` şeklindedir.

**Neden.** torch, PyTorch'un CPU dizininden, torchvision ise PyPI'dan gelmiştir. Bu iki derleme
birlikte yüklenemez; `pip check` hiçbir sorun bildirmese de hiçbir model dönüştürülemez.

**Çözüm.** yonga'nın çalıştığı ortamda yalnızca torchvision'ı değiştirin, sonra içe aktarmayı
denetleyin:

```bash
pip uninstall -y torchvision && pip install torchvision --index-url https://download.pytorch.org/whl/cpu
yonga doctor --full
```

Burada `pip install --force-reinstall` kullanmayın: tüm bağımlılıkları da yeniden kurar ve başka
sürüm sabitlemelerini bozabilir. yonga ortamınızı sizin yerinize değiştirmez.

### AppImage başlamıyor (FUSE) { #fuse }

**Ne görürsünüz.** AppImage, FUSE ile ilgili bir hatayla başlamaz; tipik olarak bir konteynerde ya
da en az paketle kurulmuş bir sistemde.

**Çözüm.** AppImage'ın kendini bağlamak (mount) yerine geçici bir dizine açmasını sağlayın:

```bash
APPIMAGE_EXTRACT_AND_RUN=1 yonga doctor
```

Değişken tanımlıyken `self install` da aynı şekilde çalışır. Kalıcı olması için değişkeni
kabuğunuzun başlangıç dosyasında tanımlayın.

### Derleme önbelleği büyüyüp duruyor { #compile-cache }

**Ne görürsünüz.** Zamanla azalan boş disk alanı. Geri kazanabileceğiniz alan 10 GiB'a ya da boş
alanın %25'ine ulaştığında `doctor` bunu Storage altında işaretler.

**Neden.** OpenVINO her artifact, aygıt ve OpenVINO sürümü için bir derlenmiş blob tutar; büyük
modellerin NPU ve GPU blob'ları her biri birkaç GiB'tır. Kaldırılmış artifact'lerin (`orphaned`) ve
başka bir OpenVINO sürümünün (`stale-runtime`) blob'ları bir daha hiç kullanılmaz.

**Çözüm.** Neler olduğuna bakın, temizliği önizleyin, sonra temizleyin:

```bash
yonga cache list
yonga cache prune --dry-run
yonga cache prune
```

`--older-than DAYS`, `--unattributed` ve `--all` daha fazlasını kaldırır. OpenVINO sürümleri farklı
iki yonga kurulumu tek bir önbelleği paylaşıyorsa, temizliği saklayacağınız kurulumdan yapın ya da
onlara ayrı `--cache-dir` verin. `yonga env list`, yönetilen dışa aktarma ortamlarını ve
boyutlarını gösterir.

## Hiçbiri uymuyorsa { #unknown }

`runtime.unknown_error` (çıkış kodu 99), arızanın yonga için yeni olduğu anlamına gelir. Lütfen
[GitHub](https://github.com/ooguz/yonga/issues)'da şunlarla bir konu (issue) açın:

- `--debug` ile yeniden çalıştırılmış komut;
- `yonga doctor` çıktısı;
- hata raporu paketi olarak toplanmış iş günlüğü;
- modelin depo kimliği ve revizyonu.

Paket; günlüğü, iş durumunu, manifesti ve bir ana makine raporunu geçerli dizinde tek bir arşivde
toplar. Arşivin adı `yonga-report-` ve ardından iş kimliğinin ilk sekiz karakteridir (`--output`
ile başka bir yere yazdırabilirsiniz):

```bash
yonga logs JOB_ID --bundle
```

Token'lar gizlenir; ev dizini, ana makine adı, oturum adı ve Hugging Face hesap adı anonimleştirilir.
Yine de paketi eklemeden önce okuyun. Bir güvenlik sorununu ise
[Gizlilik ve güvenlik](privacy.md#vulnerabilities) sayfasında anlatıldığı gibi gizli olarak
bildirin.
