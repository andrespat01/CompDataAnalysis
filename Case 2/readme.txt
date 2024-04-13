
Case idea: 
Opdel data i to grupper: 
- Observationer, hvor individ er puzzler 
- Observationer, hvor individ er ikke-puzzler

Hypotese:
Vi ønsker at opdele vores observationer i de to grupper. 
For hver gruppe ønsker vi at undersøge, hvilke forskellige der er imellem at være puzzler og ikke-puzzler. 
Initielt vil det være en eksplorativ analyse, hvor vi undersøger forskellige og ligheder i de to grupper i data. Dette vil inkluderer: 
- Forskellen fra phase1 og phase 2 oberservationer og evt phase 2 til phase 3
- Forskellen i svar på spørgsmål
- Forskellen i målinger fra device
- Outlier detection?

Efter det eksplorative arbejde vil vi undersøge, hvad som (hvis noget) kan definere/drive forskellen mellem de to grupper.
Ved at anvende partial least squares, da kan vi for hver gruppe undersøge, hvilke input som har højest betydning for output (samlet positiv/negativ emotions).
Her kunne vi opholde disse resultater op imod hinanden og få indsigt i, hvilke observationer som er vigtige for at forklare forskellen mellem de to grupper.



### Jonatans noter ###

#Dette kunne måske være interessant at have med: (Det er vidst også med i Lines paper)
https://bioturing.medium.com/how-to-read-pca-biplots-and-scree-plots-186246aae063
# De kan laves sådan her: # https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
# Hvis vi har makkerpar id kan vi måske få noget fortolkning ud af at plotte linje mellem makre i PCA plot, retningen af linjen siger muligvis noget om hvordan de adskiller sig fra hinanden

#https://medium.com/@pozdrawiamzuzanna/canonical-correlation-analysis-simple-explanation-and-python-example-a5b8e97648d2

OBS: Efter phase 2 har deltagere også svaret på hvor svær de synes opgaven var, det er ikke med i HR_data.csv, men det ligger i response for hver ID i phase 2...