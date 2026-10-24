# Kurulum

yonga, Linux x86_64 üzerinde çalışır ve iki biçimde gelir: başka hiçbir şey kurmayı gerektirmeyen
tek dosyalık bir AppImage ve PyPI'daki bir Python paketi. Bu sayfa makinenizin neye ihtiyaç
duyduğunu (yonga'nın denetlediği ama asla kurmadığı Intel GPU ve NPU sürücüleri dahil), iki kurulum
yolunu, yonga'nın model dönüştürmek için kurduğu yönetilen dışa aktarma ortamlarını, dosyalarını
nerede tuttuğunu ve nasıl yükseltilip kaldırılacağını anlatır. Ayrıntılı belirtim GitHub'daki
[INSTALL.md](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md) dosyasındadır (İngilizce).

## Gereksinimler { #requirements }

| | |
|---|---|
| İşletim sistemi | x86_64 üzerinde Linux. Ubuntu 26.04.1 üzerinde doğrulandı; Ubuntu 24.04 ikinci hedef, ancak henüz test edilmedi. Windows henüz desteklenmiyor. |
| İşlemci ve aygıtlar | Yalnızca CPU kullanımı için herhangi bir x86_64 işlemci. NPU için Intel Core Ultra (Meteor Lake veya daha yenisi), GPU için Intel Arc / Xe GPU. |
| AppImage | glibc 2.28 veya daha yenisi (Ubuntu 20.04, Debian 11, Fedora 29 ve sonrası). |
| pip | Python 3.11 ile 3.14 arası. |
| Bellek | En az 8 GiB; bunun altında `yonga doctor` uyarır. Orta boy bir modeli dönüştürmek 16 GiB veya daha fazlasını ister. |
| Disk | Model boyutunun birkaç katı. `yonga prepare` bir modelin neye ihtiyaç duyduğunu hesaplar ve sığmıyorsa en başta durur. |

yonga'nın sakladığı her şey ev dizininizin altına gider ve yonga'nın kendisi hiçbir zaman root
yetkisi istemez. Yalnızca aşağıdaki, kendiniz yaptığınız sürücü ve grup ayarı `sudo` gerektirir.

## Intel GPU ve NPU { #the-intel-npu }

CPU ek bir kurulum gerektirmeden çalışır. GPU ve NPU ise dağıtımınızdan gelen sürücülere ihtiyaç duyar;
bunları hiçbir Python paketi ya da AppImage taşıyamaz. Neyin eksik olduğunu `yonga doctor` söyler.

**GPU.** Intel'in hesaplama çalışma zamanı (compute runtime) gerekir: Ubuntu'da `intel-opencl-icd` paketi ya da daha yeni
bir sürüm için Intel'in kendi paket deposu.

**NPU.** OpenVINO'nun NPU'yu görebilmesi için üç koşulun sağlanması gerekir; `yonga doctor` her
birini **NPU** bölümünde raporlar:

1. **Çekirdek sürücüsü yüklü olmalı.** `intel_vpu` modülü dağıtımın çekirdeğiyle gelir. Kullanıcı
   alanı tarafı, Intel'in Level Zero NPU sürücüsüdür ve bir dağıtım paketidir (referans Ubuntu
   makinesinde `intel-level-zero-npu`). `pip install openvino` bunu kurmaz.
2. **Aygıt düğümü bulunmalı.** Çekirdek modülü `/dev/accel/accel0` düğümünü oluşturur.
3. **Kullanıcınız onu açabilmeli.** Düğüm, 660 kipiyle `render` grubuna aittir ve yeni bir
   kullanıcı hesabı genellikle bu grupta değildir.

Çalışan bir NPU, `yonga doctor` çıktısında şöyle görünür:

```text
NPU
✓  /dev/accel/accel0   accessible  group render
✓  Group               render  the current user is a member
✓  driver              1789049548
✓  compiler            524291
✓  compiler-in-plugin  available  the NPU plugin compiles graphs itself
…
```

`render` grubunda değilseniz `doctor` uyarır ve sizin çalıştırmanız için bir komut yazdırır:

```text
NPU
!  /dev/accel/accel0  not accessible  /dev/accel/accel0 is owned by group 'render' and the current
                      user is not a member; read/write access is denied.
                      → add your user to the 'render' group, then log out and back in for the new
                      membership to take effect
                      → Suggested command, to run yourself: sudo usermod -aG render $USER
                      → yonga never runs sudo and never changes packages, kernel modules, permissions or group membership: run any command yourself.
!  Group              render  the current user is not a member of 'render'
```

Düzeltmek için komutu bir kez kendiniz çalıştırın, ardından oturumu kapatıp yeniden açın; yeni bir
grup üyeliği hâlihazırda açık olan oturuma uygulanmaz:

```bash
sudo usermod -aG render $USER
```

Oturumu kapatmadan tek bir kabukta denemek için o kabukta `newgrp render` çalıştırın.

!!! note "yonga'nın yapmadıkları"
    yonga hiçbir zaman `apt`, `modprobe`, `sudo` veya `usermod` çalıştırmaz ve kendisini kurduğunuz
    ortamda asla `pip` çalıştırmaz. Komutu size söyler; çalıştırıp çalıştırmamaya siz karar
    verirsiniz. `pip` çalıştırdığı tek yer, siz planı onayladıktan sonra kendi önbellek dizini
    altında kurduğu bir [yönetilen dışa aktarma ortamıdır](#managed-export-environments).

Sağlıklı bir NPU, her modelin onun üzerinde derleneceği anlamına gelmez: NPU derleyicisi bazı
grafları reddeder. yonga bunu sürücü sorunu olarak değil, model ve derleyici sınırlaması olarak
raporlar ve başka bir aygıt önerir. Bkz. [Test edilen donanım ve modeller](hardware.md) ve
[Sorun giderme](troubleshooting.md).

## AppImage ile kurulum { #appimage }

Her [GitHub sürümü](https://github.com/ooguz/yonga/releases), `yonga-X.Y.Z-x86_64.AppImage`
dosyasını ve onun `.sha256` dosyasını içerir. İmajın içinde yonga, kendi Python'u, OpenVINO
çalışma zamanı ve `serve` ek paketi bulunur. 0.5.1 için:

```bash
curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage
curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage.sha256
sha256sum --check yonga-0.5.1-x86_64.AppImage.sha256
chmod +x yonga-0.5.1-x86_64.AppImage
./yonga-0.5.1-x86_64.AppImage self install
yonga doctor
```

İndirilen dosya henüz bir komut değildir. `self install`, AppImage'ı `~/.local/bin/yonga` konumuna
kopyalar; böylece `yonga` her kabukta çalışır. İndirdiğiniz dosyayı sonra silebilirsiniz. Size ait
başka bir dizine kurmak için `--bin-dir DIR` ekleyin.

- **PATH.** `self install`, `yonga` komutunun artık kopyayı çalıştırıp çalıştırmadığını denetler.
  `~/.local/bin` dizini `PATH` içinde değilse bunu söyler, ne yapmanız gerektiğini yazdırır ve `2`
  koduyla çıkar. Ubuntu ve Debian'da `~/.profile`, `~/.local/bin` dizinini oturum açılışında
  ekler, ama yalnızca dizin o sırada zaten varsa. Bu yüzden ilk kurulumdan sonra bir kez oturumu
  kapatıp açın ya da geçerli kabukta şunu çalıştırın:

    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

    yonga `~/.profile`, `~/.bashrc` veya başka bir başlangıç dosyasını asla düzenlemez.

- **Dokunmadıkları.** `~/.local/bin/yonga` konumunda yonga AppImage'ı olmayan her şey (pip veya
  pipx komutu, sembolik bağlantı, dizin, betik) reddedilir ve olduğu gibi bırakılır. Yazma izniniz
  olmayan bir dizin de reddedilir; `--bin-dir` ile size ait bir dizin seçin.
- **FUSE.** AppImage kendini FUSE ile bağlar. FUSE'un bulunmadığı yerlerde (kapsayıcılar, bazı
  minimal kurulumlar) `APPIMAGE_EXTRACT_AND_RUN=1` değişkenini ayarlayın; AppImage bunun yerine
  kendini geçici bir dizine açar. Bu `self install` için de geçerlidir:

    ```bash
    APPIMAGE_EXTRACT_AND_RUN=1 ./yonga-0.5.1-x86_64.AppImage self install
    ```

- **Dönüştürme.** Dönüştürme araç zinciri (optimum-intel, NNCF, Transformers, PyTorch) imajda
  yoktur: yaklaşık 1,5 GB tutar ve doğru Transformers sürümü modele bağlıdır. Bir modeli
  dönüştüren ilk `prepare`, onun için bir
  [yönetilen dışa aktarma ortamı](#managed-export-environments) kurar.

İmajdaki Python paketleri imaj derlenirken PyPI'dan çözümlenir. Bu yüzden her sürüm, paketlenen her
paketin sürümünü ve içindeki yonga wheel'inin SHA-256 değerini listeleyen
`yonga-X.Y.Z-x86_64.AppImage.packages.txt` dosyasını da içerir.

`yonga doctor`, yonga'nın nasıl çalıştığını **Host → Install** satırında gösterir ve `PATH`
üzerindeki `yonga` bir AppImage'ı çalıştırmıyorsa uyarır. `self install` ayrıntıları
[CLI_SPEC.md](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#12c-self) dosyasındadır.

## pip ile kurulum { #pip }

yonga 0.5.1'den itibaren PyPI'dadır. Bir sanal ortama kurun:

```bash
python3 -m venv ~/.venvs/yonga
source ~/.venvs/yonga/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "yonga[openvino,export,serve]"
```

`torch` satırı isteğe bağlıdır, ancak NVIDIA GPU'su olmayan bir makinede önerilir. `export` ek
paketi PyTorch'u da kurar ve Linux'ta PyPI'daki varsayılan derleme CUDA taşır: bir Intel AI PC'nin
hiç kullanmayacağı yaklaşık 4 GB'lık kitaplık, buna karşılık CPU wheel'i yaklaşık 200 MB'tır.

!!! warning "torch ve torchvision'ı birlikte kurun"
    İkisini de yukarıdaki gibi tek komutla CPU dizininden alın. Oradan yalnızca `torch` alınırsa
    pip `torchvision` paketini PyPI'dan getirir, ikisi birlikte yüklenemez ve model dönüştürme
    `operator torchvision::nms does not exist` hatasıyla başarısız olur. `yonga prepare` bunu
    hiçbir şey indirmeden önce saptar (`export.torchvision.build_mismatch`, çıkış kodu 10),
    `yonga doctor` da **OpenVINO → exporter import** satırında gösterir. Bu durumdaki bir ortamın
    onarımı
    [TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#57-a-managed-export-environment-could-not-be-built)
    dosyasında anlatılıyor.

### Ek paketler { #extras }

Ek paketleri (extras) ne yapmak istediğinize göre seçin:

| Ek paket | Kurdukları | Gerektiği yer |
|---|---|---|
| `openvino` | `openvino`, `openvino-genai`, `openvino-tokenizers` | `probe`, `chat`, `bench`, `eval`, `serve` ve `doctor`'ın aygıt ve NPU bölümleri |
| `export` | `optimum-intel[openvino]`, `nncf` (`transformers` ve `torch` paketlerini de getirir) | `prepare`, kendi ortamınızda dönüştürme |
| `serve` | `fastapi`, `uvicorn`, `pillow` | `serve`, OpenAI uyumlu HTTP API |
| `all` | `openvino`, `export` ve `serve` | bir kullanıcının çalıştırabileceği her şey |

`export` ek paketi olmadan da `prepare` çalışır: AppImage'da olduğu gibi dönüştürmeyi bir
[yönetilen dışa aktarma ortamında](#managed-export-environments) yapar. Hiçbir ek paket olmadan
(`pip install yonga`) `doctor`, `inspect`, `list`, `logs` ve `registry` çalışır; `doctor` ise `10`
koduyla çıkar ve eksik paketi ve onu sağlayan ek paketi söyler.

!!! note "OpenVINO paketleri bir takımdır"
    `openvino`, `openvino-genai` ve `openvino-tokenizers` aynı sürüm ailesinden gelmelidir
    (örneğin hepsi `2026.4`). Aileleri karıştırmak içe aktarma hatalarına ya da çökmelere yol açar,
    bu yüzden üçünü birlikte yükseltin. `yonga doctor` bunu **ABI family** satırında denetler ve
    farklıysalar kurulması gereken sürüm sabitlemelerini yazdırır. Bkz.
    [INSTALL.md](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#4-the-openvino-abi-family-rule).

pip kurulumunda `self install` yoktur: `yonga` komutu ortamın `bin` dizininde durur, kullanmak için
ortamı etkinleştirin. Bunun yerine `~/.local/bin` içinde bir `yonga` komutu istiyorsanız
`pipx install 'yonga[openvino,export]'` bunu yapar. Bu durumda PyTorch PyPI'dan gelir; bu da
Linux'ta daha büyük olan CUDA derlemesidir.

## Yönetilen dışa aktarma ortamları { #managed-export-environments }

Bir modeli dönüştürmek, o mimarinin Hugging Face Transformers'taki kodunu çalıştırır ve bazı
mimariler doğru dışa aktarma için belirli bir Transformers sürümüne ihtiyaç duyar. yonga'nın
yanındaki paketler bir modelin bu gereksinimini karşılayamadığında ya da yonga'nın yanında hiç
dönüştürme araç zinciri olmadığında (AppImage ya da `export` olmadan yapılmış bir pip kurulumu),
`prepare` durmak yerine dönüştürmeyi bir **yönetilen dışa aktarma ortamında** çalıştırır:

- Bu, `~/.cache/yonga/envs` altında, yalnızca dışa aktarmanın kullandığı yalıtılmış bir Python
  sanal ortamıdır. Kendi ortamınız hiçbir zaman değiştirilmez. AppImage, kendi Python'unun bir
  kopyasını da orada tutar, çünkü imajın bağlama noktası her çalıştırmada değişir.
- İçinde yonga'nın test edildiği araç zinciri bulunur; yalnızca gereken sürüm sabitlemesi
  değiştirilir, PyTorch CPU dizininden gelir ve OpenVINO, çalışma zamanının dışa aktarmanın
  yazdıklarını okuyabilmesi için çalışma zamanının sürümüne sabitlenir.
- Hazırlama planı bunu **Export environment** satırında duyurur ve kurmadan önce sorar (PyPI'dan ve
  PyTorch CPU dizininden en fazla yaklaşık 3 GiB indirilir). `--yes` bunu onaylar.
- Ortam, model indirilmeden önce kurulur ve doğrulanır; böylece başarısız bir kurulum size
  gigabaytlarca indirmeye değil, birkaç dakikaya mal olur. Aynı gereksinimlere sahip sonraki işler
  onu yeniden kullanır.
- Kurulum için bir kez ağ gerekir. Çevrimdışıyken mevcut bir ortam yeniden kullanılır; yeni bir
  ortama ihtiyaç duyan iş ise `export.env.offline` ile durur.

Ortamları görmek ve silmek için:

```bash
yonga env list
yonga env remove KEY
```

Silinen bir ortam, sonraki bir dışa aktarma ona ihtiyaç duyarsa yeniden kurulur. Yönetilen ortamları
tümden reddetmek için `prepare` komutuna `--no-managed-env` verin ya da `config.toml` içinde
`managed_export_envs = false` ayarlayın; hazırlama bu durumda indirmeden önce durur (çıkış kodu 10)
ve gereken paket sürümünü söyler. Ayrıntılar
[INSTALL.md](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#6-the-transformers-version-caveat)
dosyasında.

## Kurulumu denetleyin { #check }

```bash
yonga --version
yonga doctor
```

`doctor`, her şey kullanılabiliyorsa `0`, uyarılarla kullanılabiliyorsa `2`, yonga'nın aşamayacağı
bir eksik varsa `10` koduyla çıkar. Ubuntu'da RAM tabanlı küçük bir `/tmp` tek başına `2` koduna yol
açar: bu uyarı beklenen bir durumdur, çünkü yonga kendi disk tabanlı geçici dizininde dönüştürür.
Ardından [Hızlı başlangıç](getting-started.md#prepare) ile devam edin.

## Dosyalar nerede durur { #where-files-live }

yonga, XDG temel dizin düzenini izler:

| Yol | İçerik |
|---|---|
| `~/.config/yonga/config.toml` | Ayarlar (isteğe bağlı) |
| `~/.config/yonga/registry-overrides/` | Kendi uyumluluk kurallarınız |
| `~/.local/share/yonga/artifacts/` | Hazırlanmış modeller (artifact'ler), her biri bir manifest ile |
| `~/.local/share/yonga/logs/`, `benchmarks/`, `chat-history/` | İş günlükleri, performans ölçümü sonuçları, sohbet geçmişi |
| `~/.cache/yonga/tmp/` | Dönüştürmeler için yönetilen geçici dizinler |
| `~/.cache/yonga/compile/` | OpenVINO derleme önbelleği (`yonga cache list`) |
| `~/.cache/yonga/envs/` | Yönetilen dışa aktarma ortamları (`yonga env list`) |
| `~/.local/bin/yonga` | `self install`'un oluşturduğu AppImage kopyası |

`yonga registry paths`, makinenizdeki gerçek konumları yazdırır. `XDG_CONFIG_HOME`, `XDG_DATA_HOME`
ve `XDG_CACHE_HOME` değişkenleri dikkate alınır. Modelleri ve önbellekleri başka bir diskte tutmak
için dizinleri tek bir komut için, ortam değişkeniyle ya da `config.toml` içinde ayarlayın:

=== "Komut başına"

    ```bash
    yonga --data-dir /mnt/big/yonga-data --cache-dir /mnt/big/yonga-cache list
    ```

=== "Ortam değişkeni"

    ```bash
    export YONGA_DATA_DIR=/mnt/big/yonga-data
    export YONGA_CACHE_DIR=/mnt/big/yonga-cache
    ```

=== "config.toml"

    ```toml
    data_dir = "/mnt/big/yonga-data"
    cache_dir = "/mnt/big/yonga-cache"
    ```

Komut satırı seçenekleri `YONGA_*` değişkenlerinin, onlar da `config.toml` dosyasının önüne geçer.
Kaynak modeller normal Hugging Face önbelleğinde (`~/.cache/huggingface`) kalır; hiçbir zaman
kopyalanmaz ya da değiştirilmez.

## Yükseltme { #upgrading }

=== "AppImage"

    Yeni AppImage'ı indirin, doğrulayın ve `self install` komutunu ondan çalıştırın. Komut,
    `~/.local/bin` içindeki eski kopyayı her iki sürümü de göstererek sorduktan sonra değiştirir;
    `--yes` soruyu yanıtlar.

    ```bash
    ./yonga-X.Y.Z-x86_64.AppImage self install
    ```

=== "pip"

    Aynı ortamın içinde yükseltin:

    ```bash
    pip install --upgrade "yonga[openvino,export,serve]"
    ```

    Üç OpenVINO paketini yükseltirken aynı sürüm ailesinde tutun.

Ardından `yonga doctor` çalıştırın. yonga 1.0'a ulaşana kadar bir ara sürüm komut satırını,
uyumluluk kayıt defteri (registry) biçimini ve artifact manifestini değiştirebilir; nelerin değiştiğini
[değişiklik günlüğü](https://github.com/ooguz/yonga/blob/main/CHANGELOG.md) söyler. Bir OpenVINO
yükseltmesi eski derleme önbelleği girdilerini geride bırakır; `yonga cache prune` bunları siler
(bkz. [Bakım](guide/maintain.md)).

## Kaldırma { #uninstalling }

=== "AppImage"

    ```bash
    yonga self uninstall
    ```

    Bu komut, sorduktan sonra `~/.local/bin` içindeki kopyayı siler. Hazırlanmış modeller,
    günlükler, önbellekler ve ayarlar yerinde kalır; komut bunların nerede olduğunu yazdırır.

=== "pip"

    ```bash
    pip uninstall yonga
    ```

    Ya da sanal ortamın tamamını silin. Hazırlanmış modeller, günlükler, önbellekler ve ayarlar
    yerinde kalır.

Disk alanını da boşaltmak için kaldırmadan önce artifact'leri `yonga remove` ile, derleme önbelleğini
`yonga cache prune --all` ile silin ya da kaldırdıktan sonra yonga'nın dizinlerini silin:

```bash
rm -rf ~/.local/share/yonga ~/.cache/yonga ~/.config/yonga
```

!!! warning "Uyarı"
    Bu komut tüm hazırlanmış modelleri, günlükleri, performans ölçümlerini, yönetilen dışa aktarma
    ortamlarını ve ayarları siler. Dizinleri `--data-dir`, `--cache-dir` veya `YONGA_*`
    değişkenleriyle taşıdıysanız onları silin. `~/.cache/huggingface` içindeki Hugging Face
    önbelleği tüm Hugging Face araçlarına aittir ve silinmez; onu `hf cache` ile
    temizleyebilirsiniz.
