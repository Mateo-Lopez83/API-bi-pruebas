from pydantic import BaseModel
'''
- `r_standard`
- `g_standard`
- `class_GALAXY`
- `class_STAR`
- `mjd_sequential_standard`
- `month_sin_standard`
- `month_cos_standard`
- `x_cord_standard`
- `y_cord_standard`
- `z_cord_standard`
'''
class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    r: float
    g: float
    mjd: int
    ra: float
    dec: float
    clase: object

#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["r","g","mjd","ra", "dec", "class"]