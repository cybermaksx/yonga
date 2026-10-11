# Çıkış kodları

Her yonga komutu kararlı bir çıkış koduyla biter. Böylece bir betik, hiçbir metni ayrıştırmadan
"NPU bu modeli derleyemiyor" ile "disk dolu" durumlarını birbirinden ayırabilir. Bu sayfa her çıkış
kodunu ve ona genellikle yol açan hataları listeler, kodun nasıl seçildiğini açıklar ve başarısız
bir komutun `--json` ile yazdığı JSON belgesini anlatır. Kodlar
[`src/yonga/domain/diagnostics.py`](https://github.com/ooguz/yonga/blob/main/src/yonga/domain/diagnostics.py)
içindeki `ExitCode` ile tanımlanır ve
[CLI_SPEC.md bölüm 15](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#15-exit-codes)'te
belirtilir.

## Tüm kodlar { #codes }

Son sütundaki tanı kodları örnektir, tam liste değildir. Sık görülenleri
[Sorun giderme](../troubleshooting.md) sayfası açıklar; yerleşik uyumluluk kayıt defterinin
(registry) üretebileceği her kod ise
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#3-registry-issue-codes)'de
listelenir.

| Kod | Anlamı | Tipik nedenler |
|---|---|---|
| 0 | Başarılı | Komut istediğinizi yaptı. `--help` de 0 ile çıkar; çıktısını yarıda kestiğiniz bir komut da öyle (örneğin `yonga logs JOB_ID --raw \| head`). |
| 2 | Uyarılarla başarılı | `doctor` yalnızca uyarı buldu, örneğin RAM üzerinde duran küçük bir `/tmp`. `prepare` tamamlandı, ama bir geri dönüş (fallback) aygıtında ya da başka bir uyarıyla. `bench` bazı çalıştırmaları ölçtü, diğerleri başarısız oldu. `probe` geçen ya da başarısız olan bir aygıt bulamadı ve en az biri sonuçsuz kaldı (`--quick` ya da `--timeout` süresi doldu). `self install` kopyayı yerleştirdi ama `PATH` üzerindeki `yonga` henüz onu çalıştırmıyor. `registry validate`, kurulu bir uzak paketin artık doğrulanmadığını gördü. |
| 10 | Ortam kullanılamıyor | Gerekli bir paket eksik (`runtime.openvino.missing`). NPU aygıt düğümüne erişim izni yok (`npu.device.permission_denied`). Sürücü başlatılamıyor (`device.driver_missing`). Yerel kütüphaneler farklı sürümlerden (`packages.abi_mismatch`). Dışa aktarıcı içe aktarılamıyor (`export.torchvision.build_mismatch`) ya da bu Python sürümünde çalışamıyor (`export.optimum.normalized_config`). `--no-managed-env` ile karşılanmamış bir dışa aktarıcı sürüm sabitlemesi (`preflight.exporter_requirement_unsatisfied`). `doctor`, yonga'nın ihtiyaç duyduğu bir şey eksik olduğunda 10 ile çıkar. |
| 20 | Model ya da kaynak çözümlenemedi | Depo yok ya da sizin için görünür değil (`hub.repo.not_found`). `--offline` etkin ve model yerel önbellekte yok (`hub.offline.missing`). `--offline` ile `search` (`hub.offline.search_unavailable`). Mimarisi içe aktarılamayan bir GGUF deposu (`source.gguf_unsupported`). Kurulamayan ya da çevrimdışıyken ağa ihtiyaç duyan bir yönetilen dışa aktarma ortamı (`export.env.install_failed`, `export.env.offline`). |
| 21 | Kimlik doğrulama gerekli | Erişimi kısıtlı (gated) ya da özel bir depo ve geçerli bir Hugging Face erişim belirteci (token) yok (`hub.auth.gated`, `hub.auth.required`). |
| 22 | Uzak kod onayı gerekli | Model ancak deposundaki Python kodu çalıştırılarak dönüştürülebiliyor ve `--trust-remote-code` verilmedi (`source.remote_code_required`). |
| 30 | Yetersiz disk ya da bellek | Ön denetim, hiçbir şey indirilmeden işi reddetti (`preflight.insufficient_disk`). Bir yazma işlemi dolu bir dosya sistemine çarptı (`storage.insufficient_disk`, `disk.insufficient_space`). Bellek tükendi (`runtime.out_of_memory`, `memory.out_of_memory`). |
| 40 | Dönüştürme başarısız | Dışa aktarma başarısız oldu ve hiçbir kural nedenini tanımadı (`export.failed`). |
| 41 | Doğrulama başarısız | Dışa aktarılan IR geri okunamıyor (`tokenizer.ir.string_precision`, `ir.unsupported_element_type`) ya da hazırlanmış model (artifact) doğrulamayı başka bir nedenle geçemedi. |
| 42 | Onarım gerekli ama devre dışı | Doğrulama, bilinen bir onarımın düzelttiği bir sorun buldu ve `--no-repair` verilmişti (`repair.declined`). |
| 50 | İstenen aygıt desteklenmiyor | `prepare` aygıtı kullanamıyor: OpenVINO aygıtı göstermiyor (`device.not_present`), kayıt defteri aygıtın bu modeli çalıştıramayacağını biliyor (`device.known_unsupported`) ya da aygıt yoklamayı geçemedi; `--device auto` tüm geri dönüş seçeneklerini denedikten sonra da bu kod döner. |
| 51 | Derleme başarısız | Aygıtın derleyicisi modeli `probe`, `chat` ya da `bench` sırasında reddetti (`npu.compiler.unsupported_graph`, `npu.compiler.pass_failed` ve diğer `npu.compiler.*` kodları). |
| 52 | Üretim başarısız | Model yüklendikten sonra metin üretimi başarısız oldu. `bench` de hiçbir çalıştırması ölçüm üretmediğinde 52 ile çıkar. |
| 60 | İptal edildi | Ctrl-C'ye bastınız (`job.cancelled`), bir onayı reddettiniz (`job.declined`) ya da terminal ve `--yes` olmadığı için onay sorulamadı (`job.confirmation_unavailable`). |
| 70 | Kayıt defteri ya da yapılandırma hatası | Geçersiz bir seçenek değeri ya da yapılandırma dosyası, bir kayıt defteri sorunu, grup boyutu modele uymayan bir profil (`export.quantization.group_size_mismatch`), `chat` ya da `serve` ile verilen `--json` ve yonga'nın ayrıştıramadığı her komut satırı: bilinmeyen bir komut ya da seçenek, eksik bir argüman, yanlış türde bir değer ya da komutsuz `yonga`. |
| 99 | Beklenmeyen iç hata | Bir hata (`internal.unexpected_error`) ya da hiçbir kuralın tanımadığı bir arıza (`runtime.unknown_error`). Lütfen [bildirin](../troubleshooting.md#unknown). |

## Kod nasıl seçilir { #how }

Her arıza bir **kategoriye** sahip bir tanıdır ve çıkış kodunu bu kategori belirler. Birkaç kural
bunu inceltir:

- **40, 42 ve 52 bir nedeni değil, bir aşamayı adlandırır.** "Dönüştürme", "onarım" ve "üretim"
  anlamına gelirler ve yalnızca arızanın nedeni bilinmediğinde görülürler. Tanınan bir nedenle
  başarısız olan bir dönüştürme o nedenin koduyla çıkar: grup boyutu hatası 40 değil, 70 ile
  çıkar. Kod 41 hem doğrulama aşamasının kodu hem de IR ve tokenizer kategorilerinin kodudur.
- **`prepare` aygıt arızalarını 50 olarak bildirir.** Kategori ne olursa olsun, `prepare`'in
  kullanamadığı bir aygıt 50 ile biter. `probe`, `chat` ve `bench` ise kategorinin kendi kodunu
  korur; bu yüzden aynı NPU derleyici reddi `prepare`'den 50, `probe`'dan 51 olarak döner.
- **Bozuk bir ana makine her yerde 10'dur.** Aygıt düğümüne erişim izninin olmaması ya da
  başlatılamayan bir sürücü her komutta 10 ile çıkar ve `prepare` bu durumda başka bir aygıta geri
  dönmez: hangi aygıtı seçtiğinizin önemi olmadan önce ana makinenin düzeltilmesi gerekir.
- **Kullanım hataları 2 değil, 70'tir.** Çoğu komut satırı aracı bir yazım hatası için 2 ile
  çıkar. yonga 2'yi uyarılar ve sonuçsuz yoklamalar için ayırır; böylece sonuçsuz bir `probe`'u
  yeniden deneyen bir betik, bir yazım hatasını onunla karıştırmaz.

Tanının kendi kodu ve kategorisi `--json` belgesinde ve iş günlüğünde saklanır; çıkış kodu daha
genel olsa bile neden kaybolmaz.

## `--json` hata belgesi { #json }

`--json` ile başarısız olan bir komut standart çıktıya tam olarak bir JSON belgesi, standart hataya
ise süslemesiz, insan tarafından okunabilir tanıyı yazar. Bu, `yonga --offline --json search qwen`
komutunun gerçek çıktısıdır (okunması için biçimlendirildi):

```json
{
  "ok": false,
  "exit_code": 20,
  "diagnostic": {
    "code": "hub.offline.search_unavailable",
    "category": "download_failure",
    "message": "Offline mode is active, and searching the Hugging Face Hub is a network request: there is no local copy of the Hub to search.",
    "fallback_devices": [],
    "job_id": null,
    "title": "Search needs the network",
    "severity": "error",
    "remediation": [
      "Run the search without offline mode (no --offline, and YONGA_OFFLINE / HF_HUB_OFFLINE unset): yonga --json search qwen",
      "A model already in the local cache can still be inspected offline: yonga --offline inspect MODEL"
    ],
    "log_path": null
  }
}
```

| Alan | İçeriği |
|---|---|
| `ok` | Arızada `false`. Başarılı bir komutun belgesi de `ok` ile başlar ve değeri `true` olur. |
| `exit_code` | Süreç çıkış kodu, tamsayı olarak. |
| `diagnostic.code` | Kararlı tanı kodu, örneğin `hub.offline.search_unavailable`. |
| `diagnostic.category` | Çıkış kodunun türetildiği arıza kategorisi, örneğin `download_failure`. |
| `diagnostic.title`, `diagnostic.message` | Terminalin gösterdiği metnin aynısı. |
| `diagnostic.severity` | Arızada `error`. |
| `diagnostic.remediation` | Önerilen adımlar, sırasıyla. Bir komut adlandıran adım sizin seçeneklerinizi tekrarlar ve kopyalanmaya hazırdır. |
| `diagnostic.fallback_devices` | Arıza tek bir aygıtla ilgiliyse, onun yerine önerilen aygıtlar. |
| `diagnostic.job_id`, `diagnostic.log_path` | İş çalıştıran komutlarda (`prepare`, `probe`, `bench`) iş ve günlük dosyası. Diğerlerinde `null`. |

`--json` genel (global) bir seçenektir, bu yüzden komut adından önce gelir; çoğu komut onu kendi
adından sonra da kabul eder (`yonga list --json`). `chat` ve `serve` onu 70 ile reddeder.

## Betiklerde çıkış kodları { #scripts }

Bir makinenin hazır olup olmadığını, uyarıları başarı sayarak denetleyin:

```bash
yonga doctor > /dev/null
case $? in
  0) echo "hazır" ;;
  2) echo "hazır, uyarılarla" ;;
  *) echo "kullanılamıyor: nedenini görmek için 'yonga doctor' çalıştırın" ;;
esac
```

Bir arızanın tanı kodunu okuyun (bu örnek `jq` kullanır):

```bash
yonga --json --offline inspect Qwen/Qwen3-0.6B 2>/dev/null | jq -r '.diagnostic.code // "ok"'
```
