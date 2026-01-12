# Estado de Mask2Former
## Nomenclatura
* **MSDEFORM**: Atención del Pixel Decoder que incluye Mask2Former, el nombre hace referencia a Multi Scale Deformable Attention.
* **FAPN**: Pixel Decoder modificado que usamos, el nombre hace referencia a Feature Aligned Pyramid Network.
* **BRA**: Modificación que metemos en el Transformer Decoder, el nombre hace referencia a Bilevel Routing Attention, el principal comoponente de la red Biformer.
* **Masked Attention**: Principal aporte de Mask2Former, restringir el cálculo de la atención a las regiones predichas por los feature maps que genera el backbone.

## Versiones de Mask2Former que tenemos en las pruebas
Ahora mismo tenemos dos modificaciones a la red, una en el Pixel Decoder y otra en el Transformer Decoder. La modificación del Pixel Decoder es sustituir la atención MSDEFORM por la que se incluye en FAPN y la modificación al Transformer Decoder implica añadir un nuevo bloque antes del cálculo de la Masked Attention, este bloque aplica la atención BRA, con la intención de que recupere características que se le hayan podido escapar al backbone y reforzar el cálculo de la masked attention.

Esto implica que nuestras pruebas usan las siguientes configuraciones de modelos:
1. Mask2Former sin modificaciones.
2. Mask2Former con BRA
3. Mask2Former con FAPN
4. Mask2Former con BRA y FAPN

Todas las pruebas se ejecutaron 5 veces en mi PC de sobremesa del citius con una 3080TI (12GB de VRAM), FiveBillionPixels (RGB) en parches de 500x500 y un batch de 5, tuve que repetir varias pruebas que hice al principio para que quedasen todas con el mismo batch.

| Model                          | mIoU (±STD)        | fwIoU (±STD)        | mACC (±STD)        | pACC (±STD)        |
|--------------------------------|--------------------|---------------------|--------------------|--------------------|
| Mask2Former sin modificaciones | 63,04 ± 0,29       | 80,57 ± 0,30        | 73,81 ± 0,65       | 88,86 ± 0,20       |
| Mask2Former con BRA            | 63,68 ± 0,30       | 81,15 ± 0,97        | 74,71 ± 0,41       | 89,17 ± 0,59       |
| Mask2Former con FAPN           | 64,13 ± 0,51       | 81,59 ± 0,15        | 75,48 ± 0,58       | 89,46 ± 0,10       |
| Mask2Former con BRA y FAPN     | 64,05 ± 0,41       | 81,49 ± 0,60        | 75,20 ± 0,42       | 89,41 ± 0,37       |



Los resultados se pueden consultar más detalladamente en este google sheets: https://docs.google.com/spreadsheets/d/10L7KO6XoB16MYGfR-T9EJu9OvSZyi8eUYy39PtEEdQs/edit?usp=sharing

## Cosas que tengo hechas a día de hoy

* El código de Mask2Former está funcional en el CTCOMP3 (CPD del citius), hice todo de 0 para saber arreglar un warning de CUDA. Funciona con varias GPU.
* El container de Singularity con Mask2Former está funcional y sin warnings en el CESGA, solo pude comprobar que funcionase porque había mucha cola.
* Tenemos también funcional el dataset de LADOS: Aerial Imagery Dataset for Oil Spill Detection, Classification, and Localization Using Semantic Segmentation (https://zenodo.org/records/15888341)

| Model                             |  mIoU | fwIoU |  mACC |  pACC |
|-----------------------------------|-------|-------|-------|-------|
| Mask2Former con FAPN              | 83.29 | 86.88 | 88.18 | 92.99 |
| Mask2Former sin modificaciones    | 81.07 | 87.45 | 89.40 | 93.30 |
| Mask2Former con BRA y FAPN        | 74.19 | 86.71 | 85.19 | 92.88 |

**Estos resultados hacen referencia a una ejecución suelta con el dataset LADOS**

## Tareas a futuro que quedan pendientes

* Crear una versión del dataset de ríos en la que pueda tener referencias a cada punto que segmente.
* Sacar resultados para el dataset de ríos.
* Buscar más datasets a los que nos podamos comparar.