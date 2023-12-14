# Synteettinen data täydentämässä COVID-19-tautia diagnosoivien neuroverkkojen koulutusdataa - lähdekoodi
Tämä repositorio sisältää käytetyn lähdekoodin kanditutkielmalle *Synteettinen data täydentämässä COVID-19-tautia diagnosoivien neuroverkkojen koulutusdataa* (Helsingin yliopisto, syksy 2023).

## Sisältö
Python-ohjelmat kannattaa suorittaa `coviddcgan` hakemiston sisältä.

### coviddcgan/coviddcgan.py
Lähde: [https://gist.github.com/anilsathyan7/ffb35601483ac46bd72790fde55f5c04](https://github.com/eriklindernoren/Keras-GAN)

Luo DCGAN-mallin ja tallentaa generoivan ja erottelevan verkon rakenteet ja painot hakemistoon `dcgan/saved_model`. Hakemistoon tallentuu myös mallien varmuuskopiot. Valmis malli kannattaa siirtää `covidmodel` (COVID-positiivisia röntgenkuvia luova malli) tai `normalmodel` (terveitä röntgenkuvia luova malli) hakemistoon koulutuksen valmistuttua, sillä uusi koulutus korvaa hakemistossa olevat mallit.

### coviddcgan/coviddensenet.py
Lähde: [https://www.kaggle.com/code/jutrera/training-a-densenet-for-the-stanford-car-dataset](https://www.kaggle.com/code/jutrera/training-a-densenet-for-the-stanford-car-dataset)

Luo DenseNet-121-mallin ja tallentaa sen rakenteen ja painot hakemistoon `dcgan/saved_densenet`. Valmis malli kannattaa siirtää `augmented_dataset` (DCGAN-mallilla täydennetyllä tietoaineistolla koulutettu malli) tai `real_dataset` (täysin todellisella tietoaineistolla koulutettu malli) hakemistoon koulutuksen valmistuttua, sillä uusi koulutus korvaa `saved_densenet` hakemistossa olevan mallin.

### coviddcgan/create_densenet_stats.py
Tulostaa DenseNet-121-mallin suorituskyvystä kertovat tilastot ja luo niistä MatPlotLib-kirjaston luomat laatikkokuvat.

### coviddcgan/dataconvert.py
Lähde: [https://gist.github.com/anilsathyan7/ffb35601483ac46bd72790fde55f5c04](https://gist.github.com/anilsathyan7/ffb35601483ac46bd72790fde55f5c04)

Muuntaa `test` ja `train` hakemistojen sisällä olevat kuvatietoaineistot NPZ-hakemistojen sisälle tallennetuiksi NPY-tiedostoiksi. Näitä voidaan käyttää koneoppimismallien koulutukseen.

### coviddcgan/generate_xrays.py
Tallentaa valitun mallin luomat röntgenkuvat NPZ-hakemiston sisälle tallennetuiksi NPY-tiedostoiksi. Halutessaan luodut röntgenkuvat voi myös tallentaa `gen_imgs` hakemistoon PNG-kuvatiedostoina.

### coviddcgan/plot_model.py
Luo valituista malleista niiden arkkitehtuuria kuvaavan PNG-kuvan.
