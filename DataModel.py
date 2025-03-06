from pydantic import BaseModel

class DataModel(BaseModel):
    
    objid: int
    ra: float
    dec: float
    u: float
    g: float
    r: float
    i: float
    z: float
    run: int
    camcol: int
    field: int
    score: float
    clean: int
    classStr: object
    mjd: int
    rowv: float
    colv: float
    
    def columns(self):
        return ["objid","ra","dec","u","g","r","i","z","run","camcol","field","score","clean","class","mjd","rowv","colv"]