# satellite.py
import numpy as np
from sgp4.api import Satrec, WGS72, jday
from astropy.time import Time, TimeDelta
from astropy import units as u
from closeApproach import CElement
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

class CSatellite:
    def __init__(self, str1:str, str2:str, str3:str):
        self.m_gStr1 = str1.strip()
        self.m_gStr2 = str2.strip()
        self.m_gStr3 = str3.strip()
        self.satrec = Satrec.twoline2rv(self.m_gStr2, self.m_gStr3, WGS72)

    def __del__(self):
        pass

    def GetSatName(self):
        # 取第一行第三位开始的字符串作为卫星名称
        Name = self.m_gStr1[2:].strip()
        return Name
    
    def GetSatID(self):
        # 取第二行前两位开始的字符串作为卫星ID
        ID = self.m_gStr2[2:7].strip()
        return ID

    def _tle_to_epoch(self):
        year = int(self.m_gStr2[18:20])
        year += 2000 if year < 57 else 1900
        day_of_year = float(self.m_gStr2[20:32])
        date = Time(f"{year}-01-01T00:00:00", scale='utc') + TimeDelta(day_of_year-1, format='jd')
        return date

    def _state_at_jd(self, jd):
        e, r, v = self.satrec.sgp4(jd, 0.0)
        if e != 0:
            raise RuntimeError(f"sgp4 error {e}")
        return np.array(r), np.array(v)

    def GetIniOrb(self):
        epoch = self._tle_to_epoch()
        jd = epoch.jd
        r, v = self._state_at_jd(jd)
        element = CElement()
        element.SetData(epoch, r, v)
        return element

    def Propagate(self, Tf):
        if isinstance(Tf, (int, float)):
            span_sec = float(Tf)
            epoch = self._tle_to_epoch()
            target_time = epoch + TimeDelta(span_sec, format='sec')
        else:
            target_time = Tf
            epoch = self._tle_to_epoch()
            span_sec = (target_time - epoch).to(u.s).value

        t = target_time
        jd = t.jd
        r, v = self._state_at_jd(jd)
        orb = CElement()
        orb.SetData(target_time, r, v)
        return orb