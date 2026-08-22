# Correr esp32-ai en Windows (WSL2 + usbipd)

`scripts/deploy.sh` da por hecho un anfitrión Unix con la placa conectada
directamente — `PORT` apunta por defecto a `/dev/cu.usbmodem*`. En Windows la
placa es invisible para las herramientas de Linux hasta que se reenvía a WSL2, y
el texto generado sale por un conector USB-C **distinto** al que se usa para
grabar.

Esta guía cubre las dos cosas. Probado en Windows 11 + WSL2 (Ubuntu), una
ESP32-S3 DevKitC-1 N16R8 y el modelo `tinystories`.

---

## Lo primero: la placa tiene dos puertos USB-C y no son intercambiables

Es lo que más tiempo cuesta, así que va de primero.

| Serigrafía | Chip | ID USB | Sirve para |
| ---------- | ---- | ------ | ---------- |
| `UART` | puente CH343 | `1a86:55d3` | **Grabar** |
| `USB`  | USB-Serial-JTAG nativo | `303a:1001` | **Ver el texto generado** |

El firmware se compila con `CDCOnBoot=cdc` y `USBMode=hwcdc` (mira el `FQBN` en
`scripts/deploy.sh`), así que `Serial` queda enlazado al periférico
USB-Serial-JTAG del chip. El puente CH343 cuelga de UART0, que `Serial` ya no usa.

La trampa es que el puerto equivocado **no se queda mudo**. El bootloader de la
ROM siempre escribe por UART0, así que el conector `UART` imprime:

```
ESP-ROM:esp32s3-20210327
rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)
SPIWP:0xee
```

Salen letras, el puerto parece el correcto, y el cuento no llega nunca.

**Graba por `UART`. Escucha por `USB`.**

---

## Requisitos

En Windows:

- WSL2 con una distribución de Linux
- [usbipd-win](https://github.com/dorssel/usbipd-win) — reenvía dispositivos USB a WSL2

Dentro de WSL2, lo habitual del proyecto: `git`, `build-essential`, `uv`, `hf` y
`arduino-cli` con el core `esp32:esp32`.

---

## ¿Sin pantalla OLED? Pon `USE_DISPLAY 0` antes de nada

El firmware por defecto maneja un panel SSD1306/SH110X. Sin panel en el bus,
`display_begin()` se queda bloqueado y la placa nunca llega a generar — parece
que la grabación falló, cuando en realidad solo falta un periférico.

En `firmware/esp32_tinystories/esp32_tinystories.ino`:

```c
#define USE_DISPLAY 0
```

La salida por puerto serie funciona perfectamente sin ningún panel.

---

## Paso 1 — Grabar (a través de WSL2)

Conecta el cable al puerto **`UART`**.

En PowerShell **como administrador**:

```powershell
usbipd list                          # anota el BUSID de 1a86:55d3
usbipd bind   --busid <BUSID>        # una vez por dispositivo; queda guardado
usbipd attach --wsl --busid <BUSID>
```

Y en WSL2:

```bash
cd ~/esp32-ai
ls /dev/ttyACM*                      # confirma que la placa apareció
PORT=/dev/ttyACM0 scripts/deploy.sh tinystories
```

`deploy.sh` imprime la huella de lo que grabó:

```
expect  : fp=a9bdd778  bytes=14912348
```

Guárdala. La placa imprime ese mismo par al arrancar y los dos deben coincidir.

> `usbipd attach` no sobrevive a desconectar la placa ni a reiniciar Windows, y
> el BUSID puede cambiar entre conexiones. Corre `usbipd list` antes de cada
> attach.

---

## Paso 2 — Ver el texto (desde Windows, no desde WSL2)

Pasa el cable al puerto **`USB`**.

Es tentador reutilizar el camino de WSL2 y correr `arduino-cli monitor`. No lo
hagas: te vas a perder el principio de cada corrida, y ese principio no se
recupera.

**Por qué.** Al pulsar `RST` la placa se desconecta del USB. Rehacer el camino
`placa -> Windows -> usbipd -> vhci_hcd -> /dev/ttyACM0` tardó unos **6 segundos
medidos** en nuestro montaje:

```
=== /dev/ttyACM0 se desconecto (10:45:38), esperando ===
cat: /dev/ttyACM0: No such device          <- se repite unos 6 segundos
=== reconectado a /dev/ttyACM0 (10:45:44) ===
 to touch it, but she was too small.       <- ya va por la mitad
```

La placa empieza a generar en menos de 5 segundos, así que el monitor siempre
vuelve tarde. Y lo perdido no se recupera: `emit()` en el sketch comprueba
`Serial.availableForWrite()` y **descarta el token** cuando no hay nadie vaciando
el buffer USB-CDC, en vez de frenar la generación. No hay nada guardado para el
que llega tarde.

Leer el puerto COM directamente desde Windows reenumera en 1–2 segundos, que
alcanza para atrapar el arranque.

```powershell
# primero devuélvele la placa a Windows
usbipd detach --busid <BUSID>
```

Windows la expone entonces como puerto COM (`Administrador de dispositivos` ->
`Puertos`, o `[System.IO.Ports.SerialPort]::GetPortNames()`), y sirve cualquier
terminal que reabra el puerto al desconectarse. En `tools/windows/` de este
repositorio hay un script de PowerShell que hace el detach, encuentra el puerto
por VID/PID y reconecta solo después de cada reset.

Una captura correcta empieza en el banner de la ROM y llega hasta el perfil:

```
=== ESP32-S3 PLE TinyLM ===
model: Vin=32768 Vout=25353 D=96 L=6 H=4 F=66 P=128  (mapped 15.6 MB)
build: bytes=14912348 fp=a9bdd778 sram=29320B psram=4.19MB
>>> Once upon a time, there was a little girl named Lily...
--- 200 tokens in 19.35 s ---
throughput: 10.34 tok/s   (94.7 ms/token)
```

La generación ocurre **una sola vez por arranque** (`loop()` solo espera), así que
cada cuento nuevo necesita un `RST`.

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
| ------- | ----- | -------- |
| Sale el banner de la ROM pero nunca el cuento | Cable en `UART` | Pásalo a `USB` |
| Monitor conectado y no llega nada | La placa ya terminó | Pulsa `RST` |
| El cuento siempre empieza a media frase | Estás escuchando por WSL2 | Lee el COM desde Windows |
| Arranca, imprime la cabecera y se cuelga | `USE_DISPLAY 1` sin panel | Ponlo en `0` |
| No aparece el COM tras el `detach` | El servicio usbipd está parado | `Start-Service usbipd` y repite el detach |
| `deploy.sh`: no encuentra el puerto | No está conectada a WSL2 | Repite `usbipd list` y `attach` |

Una trampa que vale la pena nombrar: cerrar todos los procesos llamados `usbipd`
también detiene el **servicio** de Windows, y entonces un `usbipd detach` falla en
silencio — la placa se queda tomada por WSL2 y Windows no le asigna ningún COM.
Arranca el servicio antes de hacer el detach.

---

## Una medición sobre la pantalla OLED

`RESULTS.md` publica 9,88 tok/s de extremo a extremo con el panel conectado.
Corriendo el mismo firmware con `USE_DISPLAY 0` y sin panel, medimos:

| | Publicado (con OLED) | Medido (sin OLED) |
| --- | ---: | ---: |
| Cálculo | 94,9 ms/token | 94,7 ms/token |
| Extremo a extremo | 9,88 tok/s | 10,34 tok/s |

El tiempo de cálculo no cambia — ese 0,2% es ruido de medición y el modelo corre
exactamente igual de rápido. La diferencia de ~0,46 tok/s (~4,5%) es, por lo
tanto, el costo de refrescar la pantalla, **no una optimización**.

*Advertencia: es una sola captura limpia de extremo a extremo. El tiempo de
cálculo por token sí fue estable entre corridas (19,34 s y 19,35 s para 200
tokens); el número de extremo a extremo ganaría con más repeticiones.*

---

## Créditos

El modelo, la arquitectura y este repositorio son obra de
[slvDev](https://github.com/slvDev) (Viacheslav Sierbov), bajo licencia MIT. Las
Per-Layer Embeddings son un diseño de Google, de Gemma 3n; TinyStories es de
Eldan y Li, Microsoft Research.

Esta guía aporta únicamente el camino de Windows: el hallazgo de los dos
conectores, la medición de la latencia de usbipd y la solución de leer el COM
directo. La escribió [MahomerLeon](https://github.com/MahomerLeon) a partir de un
despliegue real hecho por primera vez, con depuración incluida.

### Escrito con Claude

Depurado y escrito con [Claude](https://claude.ai) (Anthropic). Cada comando se
ejecutó sobre hardware real y cada número está medido, no estimado.
