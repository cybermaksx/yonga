# Bakım

Hazırlanmış modeller, derlenmiş dosyalar, dışa aktarma ortamları ve iş günlükleri disk alanı kaplar; yonga bunların hiçbirini kendiliğinden silmez. Bu sayfa nelerin saklandığını gösteren ve alanı güvenle boşaltan komutları, ana makinenin sağlık denetimini, AppImage'ın `PATH` üzerinde tutulmasını, uyumluluk kayıt defterinin (registry) bakımını ve yonga'nın yükseltilmesini anlatır. `remove`, `cache prune`, `env remove` ve `self uninstall` neyi sileceklerini önce gösterir ve onayınızı ister.

## Hazırlanmış modelleri görün { #see-what-is-prepared }

Bu makinedeki hazırlanmış modelleri (artifact) listelemek için şunu çalıştırın:

```bash
yonga list
```

```text
Artifact          Model            Rev      Profile            Dev     Size  Benchmark   Ready
──────────────────────────────────────────────────────────────────────────────────────────────
cc24636a2f49356a  Qwen/Qwen3-0.6B  c1899de  npu-int4-sym-g128  NPU  387 MiB  2026-10-11  ✓ yes
```

İlk sütun **artifact kimliğidir**: `chat`, `bench`, `probe`, `serve`, `eval`, `publish` ve `remove` komutlarına verdiğiniz 16 karakterlik ad. Kimliğin en az altı karakterlik benzersiz bir öneki de iş görür.

Başarısız olan ya da yarıda kesilen bir `prepare`, geride bir çalışma dizini bırakabilir (hazırlık alanında yarım kalmış bir dışa aktarma ya da yönetilen geçici dizinde dosyalar). yonga, yeniden denemede tamamlanmış bir dışa aktarmayı kullanabilmek için en yenilerini saklar; geri kazanılabilecek alan olduğunda da `yonga list` bunu tek satırla bildirir. Eskimiş olanları kaldırmak için şunu çalıştırın:

```bash
yonga list --clean
```

Aynı temizlik, yeni bir `prepare`, `probe` ya da `bench` işi her başladığında kendiliğinden de çalışır. Hâlâ çalışan bir işin dizinlerine asla dokunmaz. Ne kadar süre ve kaç dizin saklanacağını yapılandırma dosyasındaki `workspace_retention_hours` (72) ve `workspace_retention_max` (3) belirler.

## Bir artifact'i kaldırın { #remove-an-artifact }

Hazırlanmış bir modeli silmek için artifact kimliğini verin:

```bash
yonga remove cc24636a2f49356a
```

yonga planı gösterir ve bir şey silmeden önce sorar:

```text
Remove cc24636a2f49356a
  Path   …/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--npu-int4-sym-g128
  Frees  559 MiB
  Cache  …/compile/1bccad8a2629cc22
The Hugging Face download cache is never touched.
```

Artifact dizinini ve ona ait derleme önbelleği girdilerini kaldırır, başka hiçbir şeyi değil. İndirdiğiniz kaynak model Hugging Face önbelleğinde kalır.

Artifact'i depo kimliğiyle (`yonga remove Qwen/Qwen3-0.6B`) ya da dizin yoluyla da belirtebilirsiniz. Silme işlemi geri alınamadığı için verdiğiniz değer **tam olarak bir** artifact ile eşleşmelidir: bir depo kimliği ya da kimlik öneki birden çok artifact ile eşleşirse (örneğin aynı model hem NPU hem GPU için hazırlanmışsa) yonga hiçbir şey silmez, her adayı profili ve aygıtıyla listeler ve `--yes` verilmiş olsa bile 70 çıkış koduyla sonlanır. Kastettiğiniz artifact'in kimliğini verin.

| Durum | Ne olur | Çıkış kodu |
|---|---|---|
| Evet yanıtı verirsiniz ya da `--yes` geçersiniz | Artifact ve derleme önbelleği girdileri silinir | 0 |
| Hayır yanıtı verirsiniz | Hiçbir şey silinmez | 60 |
| Soru sorulacak bir terminal yok (betik, boru hattı) ve `--yes` verilmemiş | Hiçbir şey silinmez | 60 |
| Verilen değer birden çok artifact ile eşleşiyor ya da hiçbiriyle eşleşmiyor | Hiçbir şey silinmez; adaylar ya da `yonga list` gösterilir | 70 |

!!! tip "İpucu"
    `chat`, `bench`, `probe`, `eval` ve `publish` bir artifact'i yalnızca okur; bu yüzden bir depo kimliği birden çok artifact ile eşleştiğinde en yeni hazır olanı seçer ve bunu söyler. Tek ve tam eşleşmede ısrar eden yalnızca `remove` komutudur.

## Derleme önbelleği { #the-compile-cache }

OpenVINO bir modeli bir aygıt için derlediğinde yonga, derlenmiş dosyaları derleme önbelleğinde saklar (varsayılan olarak `~/.cache/yonga/compile/`); böylece o artifact ve aygıt üzerindeki bir sonraki `chat`, `serve` ya da `bench` hızlı yüklenir. Her artifact, aygıt, OpenVINO sürümü ve çalışma zamanı özellikleri bileşimi için bir girdi vardır.

Girdileri görmek için şunu çalıştırın:

```bash
yonga cache list
```

```text
Compile cache  …/cache/compile
Key                  Size  Last used   Dev  OpenVINO  Owner             Status    Model
─────────────────────────────────────────────────────────────────────────────────────────────────
1bccad8a2629cc22  172 MiB  2026-10-11  NPU  2026.4.1  cc24636a2f49356a  ✓ in-use  Qwen/Qwen3-0.6B
1 entry, 172 MiB in total (1 in-use).
```

Tek başına `yonga cache` da aynı işi yapar. Her girdinin dört durumdan biri vardır:

| Durum | Anlamı |
|---|---|
| `in-use` | Sahibi olan artifact duruyor ve kurulu OpenVINO bu dosyaları yeniden kullanabilir. |
| `orphaned` | Sahibi olan artifact artık yok: elle silinmiş ya da girdi başka bir `--data-dir` ile oluşturulmuş. |
| `stale-runtime` | Şu anda çalışandan farklı bir OpenVINO sürümüyle derlenmiş, bu yüzden asla yeniden kullanılamaz. Bir OpenVINO yükseltmesinin geride bıraktığı budur. |
| `unattributed` | Sahiplik kaydı yok ve bilinen hiçbir artifact bu anahtarı üretmiyor. Genellikle sahiplik kaydı yazmayan yonga 0.3.0 ve öncesinin oluşturduğu bir girdidir. |

### Önbelleği budayın { #prune-the-cache }

`orphaned` ve `stale-runtime` girdileri kaldırmak için şunu çalıştırın:

```bash
yonga cache prune
```

Planı (her girdi, neden seçildiği ve toplam boyut) yazdırır ve silmeden önce sorar. Aşağıdaki seçenekler seçimi genişletir ya da denetler:

| Seçenek | Etkisi |
|---|---|
| `--older-than DAYS` | Bu kadar gündür kullanılmayan her girdiyi de kaldırır. |
| `--unattributed` | Hiçbir artifact'e bağlanamayan girdileri de kaldırır. |
| `--all` | Derleme önbelleğinin tamamını boşaltır. |
| `--dry-run` | Planı gösterir; hiçbir şey silmez. |
| `--yes`, `-y` | Onay istemez. |
| `--json` | Sonucu JSON olarak verir. |

Aylık bir temizliğin ne kadar alan açacağını önceden görmek için bunları birleştirin:

```bash
yonga cache prune --older-than 30 --dry-run
```

Budama yalnızca derleme önbelleği dizinine dokunur. Çalışan bir `chat` ya da `serve` oturumunun kullandığı bir girdiyi silmenin bedeli, o modelin bir sonraki yüklemede yeniden derlenmesidir, o kadar. `--all` vermediyseniz, siz soruyu yanıtlarken bir şeyin kullanmaya başladığı girdi atlanır ve bildirilir.

`cache prune` başarıda (budanacak bir şey olmadığında ve kuru çalıştırmada da) 0, bazı girdiler kaldırılamadığında 2, onayı reddettiğinizde ya da soru sorulacak bir terminal olmayıp `--yes` de verilmediğinde 60, geçersiz bir `--older-than` değerinde 70 çıkış koduyla sonlanır.

!!! note "Not"
    `yonga doctor`, derleme önbelleğinin boyutunu ve ne kadarının geri kazanılabileceğini bildirir; geri kazanılabilir kısım 10 GiB'a ya da önbellek dosya sistemindeki boş alanın dörtte birine (en az 1 GiB) ulaştığında bu satır uyarıya dönüşür.

Ayrıntılar: [CLI_SPEC.md, bölüm 12b](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#12b-cache).

## Yönetilen dışa aktarma ortamları { #managed-export-environments }

yonga'nın birlikte çalıştığı paketler bir modeli dışa aktaramadığında (örneğin mimari farklı bir Transformers sürümü gerektirdiğinde) `prepare`, siz hazırlama planında onayladıktan sonra dışa aktarma için ayrı, yalıtılmış bir ortam kurar. Dışa aktarma araç zinciri taşımayan AppImage ilk dönüştürmesinde böyle bir ortam kurar. Bu ortamlar önbellek dizininin altında (`~/.cache/yonga/envs/`) durur ve sonraki işlerde yeniden kullanılır. Ne zaman kurulduklarını [Model hazırlama](prepare.md) sayfası anlatır.

Ortamları ve kapladıkları alanı görmek için şunu çalıştırın:

```bash
yonga env list
```

```text
fbfac3ac91ba    1.4 GiB  2026-10-11  …/cache/envs/export-fbfac3ac91ba
    nncf==3.4.0, openvino==2026.4.1, openvino-tokenizers==2026.4.1.0, optimum==2.3.0,
    optimum-intel==2.2.0, torch==2.14.0, torchvision==0.29.0, transformers==5.5.4
base Python      98 MiB  …/cache/envs/python-3.14.7-3d38950fe1ce  (the environments' base)
1 environment, 1.5 GiB in total.
The base Python is a copy of the AppImage's interpreter; it is removed with the last environment
made from it.
```

Birini silmek için anahtarını ya da anahtarın benzersiz bir önekini verin:

```bash
yonga env remove fbfac3ac91ba
```

Önce sorar (`--yes` sizin yerinize yanıtlar; terminal yoksa ve `--yes` verilmemişse hiçbir şey silmez ve 60 çıkış koduyla sonlanır). Bir ortamı kaldırmak, onunla dışa aktarılmış artifact'leri etkilemez. Sonraki bir dışa aktarma ona yeniden ihtiyaç duyarsa yonga ortamı yeniden kurar.

## İş günlükleri { #job-logs }

Her `prepare`, `probe` ve `bench` çalıştırması bir iştir ve her iş, veri dizininin altına (`~/.local/share/yonga/logs/`) JSON Lines biçiminde bir günlük yazar. En son 20 işi listelemek için şunu çalıştırın:

```bash
yonga logs
```

```text
Job logs in …/logs
job       started           command  model             device  status    took     log
─────────────────────────────────────────────────────────────────────────────────────
9dabb616  2026-10-11 20:03  bench    cc24636a2f49356a  —       ✓ ok    48.5 s   888 B
de862650  2026-10-11 20:01  prepare  Qwen/Qwen3-0.6B   NPU     ✓ ok    1m 09s  88 KiB
Start times are local, UTC+03:00.
```

Bir işin sonucu `ok`, `failed`, `cancelled`, `running` ya da `unfinished` olur. Tek bir işi okumak için kimliğini ya da benzersiz bir önekini verin:

```bash
yonga logs 9dab
```

```text
Job 9dabb616-2dda-4418-9c9d-16e7f3c42a34
(…/logs/9dabb616-2dda-4418-9c9d-16e7f3c42a34.jsonl)
  bench cc24636a2f49356a  ✓ ok, took 48.5 s
  Started 2026-10-11 20:03:11 UTC+03:00; the times below are local.
  20:03:11.697  info    job.created                job created
  20:04:00.231  info    bench.saved                benchmark result saved
  20:04:00.231  info    job.completed              job completed
```

JSON Lines dosyasını diskte durduğu gibi, her satırda bir JSON nesnesi olarak `jq` ya da bir metin düzenleyici için yazdırmak üzere `--raw` ekleyin:

```bash
yonga logs 9dab --raw
```

Bir hata bildirimine günlük eklemek için `--bundle` kullanın: iş günlüğünü, iş durumunu, artifact manifestini ve bir ana makine raporunu içeren, hassas bilgileri ayıklanmış ve anonimleştirilmiş bir `.tar.gz` yazar. `--output FILE` vermezseniz dosya, geçerli dizine `yonga-report-<job>.tar.gz` adıyla yazılır:

```bash
yonga logs 9dab --bundle
```

Başarısız bir işin günlüğünün nasıl okunacağını [Sorun giderme](../troubleshooting.md) sayfası anlatır.

## Ana makineyi doctor ile denetleyin { #check-the-host-with-doctor }

`yonga doctor`, yonga'nın bağlı olduğu her şeyi denetler ve her uyarı için bir çözüm önerir. Kurulumdan sonra, bir sürücü ya da çekirdek güncellemesinden sonra ve bir şey çalışmayı bıraktığında çalıştırın:

```bash
yonga doctor
```

Rapor şu bölümlerden oluşur:

| Bölüm | Neyi denetler |
|---|---|
| Host | İşletim sistemi, çekirdek, Python, yonga'nın nasıl kurulduğu (AppImage ya da Python ortamı ve `PATH` üzerindeki `yonga` komutunun neyi çalıştırdığı), CPU, RAM ve takas alanı. |
| OpenVINO | OpenVINO çalışma zamanı, GenAI ve Tokenizers sürümleri ve bunların aynı sürüme ait olup olmadığı (ABI ailesi); ayrıca dışa aktarma araç zinciri (`optimum-intel`, `nncf`, `transformers`) ve dışa aktarıcının deneme amaçlı içe aktarımı. AppImage'da araç zinciri, dışa aktarmaların yönetilen bir ortamda çalıştığını söyleyen tek bir satırdır. |
| Devices | Hangi OpenVINO aygıtlarının (CPU, GPU, NPU) bulunduğu ve kullanılabildiği. |
| NPU | NPU aygıt düğümü (`/dev/accel/accel0`) ve ona erişiminiz, sürücü ve derleyici sürümleri ve derleyici kitaplıkları. |
| Storage | Artifact'lerin ve önbelleklerin bulunduğu yerdeki boş alan, sistem geçici dizini, yönetilen geçici dizin, dışa aktarma ortamları ve derleme önbelleği. |
| Hugging Face | Oturum açıp açmadığınız (yalnızca erişimi kısıtlı (gated) ya da özel modeller için gerekir), Hugging Face önbellek dizini ve çevrimdışı mod. |
| Registry | Yalnızca `--full` ile: kural sayısı, yüklenen katmanlar ve doğrulama sonucu. |

`--full` ayrıca geçen sefer başarısız olan dışa aktarıcı içe aktarma denetimini, kaydedilen sonucu bildirmek yerine yeniden yapar ve ölçtüğü diğer tüm dosya sistemlerini listeler. `--json` raporun tamamını tek bir JSON belgesi olarak verir.

Referans makineden bir kesit:

```text
Devices
✓  CPU  Intel(R) Core(TM) Ultra 7 268V  intel64
✓  GPU  Intel(R) Arc(TM) Graphics (iGPU)  GPU: vendor=0x8086 arch=v20.4.4
✓  NPU  Intel(R) AI Boost  4000

NPU
✓  /dev/accel/accel0   accessible  group render
✓  Group               render  the current user is a member
…
Usable, with warnings above.
```

Çıkış kodu, bir betiğe ana makinenin model çalıştırıp çalıştıramayacağını söyler:

| Çıkış kodu | Anlamı |
|---|---|
| 0 | Kullanılabilir bir OpenVINO çalışma zamanı ve aygıt var; ilgilenilmesi gereken bir şey yok. |
| 2 | Kullanılabilir, ancak uyarılar var. Tek başına küçük, RAM üzerinde duran bir `/tmp` buna yol açar; yonga kendi diskteki geçici dizinini kullanarak bunu aşar, sizin bir şey yapmanız gerekmez. |
| 10 | Kullanılabilir bir OpenVINO çalışma zamanı ya da aygıt yok. Son satır nedenini söyler: çalışma zamanı kurulu değil (`pip install` komutuyla birlikte), içe aktarılamıyor ya da hiçbir aygıt bulunamadı. |

Sık görülen uyarıları [Sorun giderme](../troubleshooting.md), NPU sürücüsü ve izin kurulumunu [Kurulum](../installation.md) sayfası anlatır.

## AppImage'ı PATH üzerine koyun { #put-the-appimage-on-your-path }

AppImage kendini `yonga` komutu olarak kurabilir. İndirdiğiniz dosyadan şunu çalıştırın:

```bash
chmod +x yonga-0.5.1-x86_64.AppImage
./yonga-0.5.1-x86_64.AppImage self install
```

AppImage'ı `~/.local/bin/yonga` konumuna kopyalar (başka bir dizin için `--bin-dir DIR`) ve `PATH` üzerindeki `yonga` komutunun artık bu kopyayı çalıştırıp çalıştırmadığını denetler. Çalıştırmıyorsa ne yapmanız gerektiğini söyler ve 2 çıkış koduyla sonlanır: Ubuntu ve Debian'da ilk kurulumdan sonra bir kez oturumu kapatıp yeniden açın ya da geçerli kabukta `export PATH="$HOME/.local/bin:$PATH"` komutunu çalıştırın. yonga kabuk başlangıç dosyalarınızı asla düzenlemez ve asla `sudo` kullanmaz.

O konumdaki daha eski bir yonga AppImage'ını sorduktan sonra değiştirir (`--yes` sizin yerinize yanıtlar); AppImage olmayan hiçbir şeyin, örneğin bir `pip` ya da `pipx` komutunun, sembolik bağlantının veya dizinin üzerine yazmayı reddeder (çıkış kodu 70). İndirdiğiniz dosyayı sonra silebilirsiniz.

`PATH` üzerinden yeniden kaldırmak için şunu çalıştırın:

```bash
yonga self uninstall
```

Bu yalnızca `~/.local/bin/yonga` konumundaki dosyayı kaldırır. Hazırlanmış modelleriniz, günlükleriniz, önbellekleriniz ve ayarlarınız yerinde kalır; komut bunların dizinlerini yazdırır. O alanı da boşaltmak için `yonga remove`, `yonga cache prune --all` komutlarını kullanın ya da dizinleri silin.

!!! note "Not"
    `self install` yalnızca yonga AppImage'dan çalışırken işe yarar. `pip` ile yapılan bir kurulumun, ortamın `bin` dizininde zaten kendi `yonga` komutu vardır; orada `self install` 70 çıkış koduyla sonlanır ve komutun nerede olduğunu söyler.

## Uyumluluk kayıt defteri { #the-compatibility-registry }

Kayıt defteri, yonga'nın hangi mimarilerin, aygıtların ve dönüştürme profillerinin çalıştığına karar verirken ve bilinen hataları tanırken kullandığı veridir. yonga ile birlikte gelir. Şu komutlar kayıt defterinin içine bakar:

| Komut | Ne için kullanılır |
|---|---|
| `yonga registry explain MODEL` | Bir modelle hangi kuralların eşleştiğini, hangilerinin neden reddedildiğini ve hangisinin kazandığını görmek için. Tek bir aygıt için açıklama almak üzere `--device npu` ekleyin. Bir karar sizi şaşırttığında ilk başvuracağınız komut budur. |
| `yonga registry profiles` | `prepare --profile` seçeneğinin kabul ettiği dönüştürme profillerini listelemek için. Tek bir aygıtı göstermek için `--device` ekleyin. |
| `yonga registry validate` | Kayıt defterinin her katmanını yükleyip tutarlı olduğunu denetlemek için. Kendi geçersiz kılma kurallarınızı yazdıktan sonra işe yarar. |
| `yonga registry paths` | Kayıt defteri katmanlarının, yapılandırmanın ve yönetilen verilerin nerede durduğunu göstermek için. |

Örneğin Qwen3-0.6B'nin hangi NPU profilini aldığını görmek için:

```bash
yonga registry explain Qwen/Qwen3-0.6B --device npu
```

```text
Profile
    Rule                Kind      Spec   Prio   Outcome
────────────────────────────────────────────────────────────────────────────────────────────────────
·   npu-int4-sym-cw     profile     20    100   matched  openvino 2026.4.1 satisfies >=2026.2,<2027;
                                                generic rule, no constraints
✓   npu-int4-sym-g128   profile     20    110   winner  openvino 2026.4.1 satisfies >=2026.2,<2027;
                                                751.6M parameters is below 5B
```

`registry validate`, kayıt defteri geçerliyse 0, geçerli olduğu hâlde kurulu bir uzak paket yok sayıldıysa 2, sorun bulursa 70 çıkış koduyla sonlanır.

### İmzalı kayıt defteri güncellemeleri { #signed-registry-updates }

Sürümler arasında yeni kayıt defteri bilgisi imzalı bir paket olarak gelebilir. Siz söylemedikçe yonga hiçbir kayıt defteri sunucusuna bağlanmaz, hiçbir imzalama anahtarıyla gelmez ve doğrulayamadığı hiçbir şeyi kurmaz. Bir yayıncıya güvenmek sizin kararınızdır: yayıncının genel anahtarını (bir `*.pub` dosyası) `~/.config/yonga/registry-keys/` dizinine kendiniz kopyalayın. Ardından:

```bash
yonga registry update --url https://example.org/yonga/bundle.json
```

İmza, aynı URL'nin sonuna `.sig` eklenerek indirilir (yalnızca HTTPS); `--file BUNDLE` diskte duran bir paketi kurar, imzası `BUNDLE.sig` değilse `--signature` ile birlikte. yonga paketi eksiksiz doğrular; imzalama anahtarını, sürümü, son geçerlilik tarihini ve eklediği ya da değiştirdiği kuralları gösterir ve siz onayladıktan sonra kurar. İmzası bozuk olan, güvenmediğiniz bir anahtarla imzalanmış, süresi dolmuş ya da kurulu olandan eski bir paket 70 çıkış koduyla reddedilir.

```bash
yonga registry status    # hangi paket kurulu, hâlâ doğrulanıyor mu?
yonga registry reset     # paketi kaldırıp yerleşik kayıt defterine dön
```

Kurulu paket, kayıt defteri her yüklendiğinde yeniden doğrulanır. Doğrulanamaz hâle gelirse (süresi dolarsa ya da diskte düzenlenirse) yonga onu yok sayar ve yerleşik kuralları kullanır; `registry status` de nedenini söyleyip 2 çıkış koduyla sonlanır. Yapılandırma dosyasında `registry_update_policy = "never"` ayarı güncellemeleri tamamen reddeder.

Ayrıntılar: [USAGE.md, bölüm 5](https://github.com/ooguz/yonga/blob/main/docs/USAGE.md#5-ask-the-registry-why) ve [REGISTRY_SPEC.md, bölüm 12.1](https://github.com/ooguz/yonga/blob/main/docs/REGISTRY_SPEC.md#121-remote-bundles-implemented).

## yonga'yı yükseltin { #upgrade-yonga }

Yükseltmeden önce [değişiklik günlüğünü](https://github.com/ooguz/yonga/blob/main/CHANGELOG.md) okuyun. yonga'nın ana sürümü 0 olduğu sürece komut satırı, kayıt defteri şeması ve artifact manifesti herhangi bir küçük sürümde (minor) değişebilir.

=== "AppImage"

    Yeni AppImage'ı [sürümler sayfasından](https://github.com/ooguz/yonga/releases) indirin ve `self install` komutunu ondan çalıştırın (`X.Y.Z` yerine yeni sürümü yazın). Komut, `~/.local/bin/yonga` konumundaki eski kopyayı değiştirmeden önce sorar ve biliyorsa iki sürümü de gösterir:

    ```bash
    chmod +x yonga-X.Y.Z-x86_64.AppImage
    ./yonga-X.Y.Z-x86_64.AppImage self install
    ```

=== "pip"

    Paketi, kurduğunuz ortamda yükseltin:

    ```bash
    pip install --upgrade "yonga[openvino,export]"
    ```

    Kurarken kullandığınız ek paket gruplarını (extras) koruyun (HTTP sunucusunu kullanıyorsanız `serve` ekleyin).

Yükseltme verilerinize dokunmaz. Verilerinize olanlar şunlardır:

- **Hazırlanmış artifact'ler** yerinde kalır. Yeni bir sürüm bir modelin varsayılan profilini değiştirdiğinde, o modeli yeniden hazırlamak eskisinin yerine geçen değil, onun yanında duran **yeni** bir artifact oluşturur; hazırlama planı bunu önceden söyler ve eskisini yeniden kullanacak `--profile` değerini belirtir. Depo kimliği verilen komutlar bundan sonra en yeni hazır artifact'i kullanır. Artık ihtiyacınız olmadığında eskisini `yonga remove` ile kaldırın.
- **Derleme önbelleği** girdilerini korur. Yükseltme yeni bir OpenVINO sürümü getirirse eski sürümün derlediği girdiler `stale-runtime` olur: yeni çalışma zamanı ilk yüklemede dosyaları yeniden derler, `yonga cache prune` de eskilerini temizler.
- **Yönetilen dışa aktarma ortamları** yerinde kalır ve `yonga env list` ile listelenir. Artık ihtiyacınız olmayanları `yonga env remove` ile kaldırın; sonraki bir dışa aktarma gerektirirse yonga ortamı yeniden kurar.
- **Yapılandırma ve iş günlükleri** olduğu gibi korunur.

Yükseltmeden sonra çalıştıracağınız `yonga doctor`, çalışma zamanının ve aygıtların hâlâ çalıştığını doğrular ve derleme önbelleğinin ne kadarının geri kazanılabileceğini bildirir.

## Neler nerede durur { #where-things-live }

| Yol | İçerik | Yönetmek için |
|---|---|---|
| `~/.local/share/yonga/artifacts/` | Hazırlanmış artifact'ler | `yonga list`, `yonga remove` |
| `~/.local/share/yonga/logs/` | İş günlükleri | `yonga logs` |
| `~/.local/share/yonga/benchmarks/` | Kaydedilmiş performans ölçümü sonuçları | `yonga bench` |
| `~/.cache/yonga/compile/` | Derleme önbelleği | `yonga cache list`, `yonga cache prune` |
| `~/.cache/yonga/envs/` | Yönetilen dışa aktarma ortamları | `yonga env list`, `yonga env remove` |
| `~/.cache/yonga/tmp/` | Yönetilen geçici dizin | `yonga list --clean` |
| `~/.config/yonga/` | `config.toml`, `registry-overrides/`, `registry-keys/` | elle |
| `~/.local/bin/yonga` | AppImage kopyası | `yonga self install`, `yonga self uninstall` |

`yonga registry paths`, `--data-dir`, `--cache-dir` ya da yapılandırma dosyasıyla değiştirdikleriniz dahil gerçek konumları yazdırır. İndirilen kaynak modeller normal Hugging Face önbelleğinde kalır; yonga onları asla kopyalamaz, değiştirmez ya da silmez.
