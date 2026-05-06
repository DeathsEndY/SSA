import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

# 常量
Rad = np.pi / 180.0
Deg = 180.0 / np.pi
GM_Earth = 398600.4418  # km^3/s^2, 地球引力常数
Pi = np.pi

class OUTMINANGLE:
    # 存储两个卫星的纬度辐角
    def __init__(self, conArgLat=0.0, tarArgLat=0.0):
        self.ConArgLat = conArgLat
        self.TarArgLat = tarArgLat

class OUTINFO:
    # 存储时间（Tend）和两个卫星的参数
    def __init__(self, Tend=None, conArgLat=0.0, tarArgLat=0.0):
        self.Tend = Tend
        self.ConArgLat = conArgLat
        self.TarArgLat = tarArgLat

class CElement:
    # 表示轨道元素，封装了轨道的状态（位置、速度、时间等）和相关计算
    def __init__(self):
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._epoch = Time.now()
        self._orbit = None

    def SetData(self, epoch: Time, r: np.ndarray, v: np.ndarray):
        # 设置轨道的初始状态
        self._epoch = epoch
        self._pos = np.asarray(r, dtype=float)
        self._vel = np.asarray(v, dtype=float)
        self._orbit = Orbit.from_vectors(Earth,
                                        self._pos * u.km,
                                        self._vel * u.km/u.s,
                                        epoch=epoch)

    def GetPos(self):
        return np.array(self._pos)

    def GetVel(self):
        return np.array(self._vel)

    def GetEpoch(self):
        return self._epoch

    def GetI(self):
        return float(self._orbit.inc.to(u.deg).value)

    def GetO(self):
        return float(self._orbit.raan.to(u.deg).value)

    def GetW(self):
        return float(self._orbit.argp.to(u.deg).value)

    def GetE(self):
        return float(self._orbit.ecc.value)

    def GetM(self):
        return float(self._orbit.nu.to(u.rad).value)  # 近似M, 仅用于兼容
    def GetA(self): return float(self._orbit.a.to(u.km).value)
    def GetTp(self): return float(self._orbit.period.to(u.s).value)

    def CalTruA(self, e=None, m=None):
        if e is None:
            e = self.GetE()
        nu = self._orbit.nu.to(u.rad).value
        return float(np.degrees(nu))
    
    def Calrp(self):
        # 计算近地点距perigee
        a = self.GetA()
        e = self.GetE()
        return a * (1 - e)
    
    def Calra(self):
        # 计算远地点距apogee
        a = self.GetA()
        e = self.GetE()
        return a * (1 + e)

    def CalTimeatU(self, u_deg, LMD_U=None, PEI_U=None, Beta=None):
        # 计算从初始状态到达到指定纬度辐角u_deg的时间
        target_u = np.deg2rad(u_deg % 360.0) # 纬度辐角
        omega = self._orbit.argp.to(u.rad).value # 近地点幅角
        target_nu = (target_u - omega) % (2*np.pi) # 纬度辐角和近地点幅角的差值，得到交线出tar星的真近点角
        if target_nu > np.pi:
            target_nu -= 2*np.pi
        e = self.GetE()
        cosE = (e + np.cos(target_nu))/(1 + e * np.cos(target_nu))
        cosE = np.clip(cosE, -1.0, 1.0)
        E = np.arccos(cosE)
        if target_nu < 0:
            E = 2*np.pi - E
        M_target = E - e*np.sin(E)
        nu0 = self._orbit.nu.to(u.rad).value
        E0 = 2*np.arctan(np.tan(nu0/2)*np.sqrt((1-e)/(1+e)))
        M0 = E0 - e*np.sin(E0)
        dM = (M_target - M0) % (2*np.pi)
        n = np.sqrt(GM_Earth / (self.GetA() ** 3))
        dt = dM / n
        # print(f"Calculated time difference (s): {dt:.3f}")
        return self._epoch + TimeDelta(dt, format='sec')

class CloseApproach:
    # 计算两个卫星的接近参数（如最小角度、最大角度、时间等）
    def __init__(self): pass

    def CalMinAngle(self, tarOrb:CElement, conOrb:CElement):
        # 计算两个卫星轨道的纬度辐角
        ConSatI = conOrb.GetI()
        ConSatO = conOrb.GetO()
        TarSatI = tarOrb.GetI()
        TarSatO = tarOrb.GetO()
        h2 = np.array([np.sin(TarSatI*Rad)*np.sin(TarSatO*Rad),
                       -np.sin(TarSatI*Rad)*np.cos(TarSatO*Rad),
                        np.cos(TarSatI*Rad)])
        h1 = np.array([np.sin(ConSatI*Rad)*np.sin(ConSatO*Rad),
                       -np.sin(ConSatI*Rad)*np.cos(ConSatO*Rad),
                        np.cos(ConSatI*Rad)])
        W = np.cross(h1,h2)
        W = W/np.linalg.norm(W)
        if W[2] < 0:
            W = - W
        nc = np.array([np.cos(ConSatO*Rad), np.sin(ConSatO*Rad), 0.0])
        nt = np.array([np.cos(TarSatO*Rad), np.sin(TarSatO*Rad), 0.0])
        ConArgLat = np.degrees(np.arccos(np.clip(np.dot(W,nc)/(np.linalg.norm(W)*np.linalg.norm(nc)),-1,1)))
        TarArgLat = np.degrees(np.arccos(np.clip(np.dot(W,nt)/(np.linalg.norm(W)*np.linalg.norm(nt)),-1,1)))
        return OUTMINANGLE(ConArgLat, TarArgLat)

    def CalMaxAngle(self, tarOrb:CElement, conOrb:CElement):
        ConSatI = conOrb.GetI(); 
        ConSatO = conOrb.GetO()
        TarSatI = tarOrb.GetI(); 
        TarSatO = tarOrb.GetO()
        h1 = np.array([np.sin(TarSatI*Rad)*np.sin(TarSatO*Rad),
                       -np.sin(TarSatI*Rad)*np.cos(TarSatO*Rad),
                        np.cos(TarSatI*Rad)])
        h2 = np.array([np.sin(ConSatI*Rad)*np.sin(ConSatO*Rad),
                       -np.sin(ConSatI*Rad)*np.cos(ConSatO*Rad),
                        np.cos(ConSatI*Rad)])
        W = np.cross(h1,h2); 
        W=W/np.linalg.norm(W)
        if W[2] > 0:
            W = - W
        nc = np.array([np.cos(ConSatO*Rad), np.sin(ConSatO*Rad),0.0])
        nt = np.array([np.cos(TarSatO*Rad), np.sin(TarSatO*Rad),0.0])
        ConArgLat = np.degrees(np.arccos(np.clip(np.dot(W,nc)/(np.linalg.norm(W)*np.linalg.norm(nc)),-1,1)))
        TarArgLat = np.degrees(np.arccos(np.clip(np.dot(W,nt)/(np.linalg.norm(W)*np.linalg.norm(nt)),-1,1)))
        return OUTMINANGLE(360-ConArgLat, 360-TarArgLat)

    def FindArgTime(self, TarSat, ConSat, Num:int):
        # 计算两个卫星在指定轨道周期后，主目标到达交线的时间
        TarIni = TarSat.GetIniOrb() # 获取目标卫星的初始轨道元素
        ConIni = ConSat.GetIniOrb() # 获取干扰卫星的初始轨道元素
        dT = (TarIni.GetEpoch() - ConIni.GetEpoch()).sec
        if dT > 0:
            T0 = TarIni.GetEpoch() + TimeDelta(Num*TarIni.GetTp(), format='sec')
        else:
            T0 = ConIni.GetEpoch() + TimeDelta(Num*TarIni.GetTp(), format='sec')
        TarIter = TarSat.Propagate(T0)
        ConIter = ConSat.Propagate(T0)
        out = self.CalMinAngle(TarIter, ConIter)
        m_TarArgLat = out.TarArgLat
        m_TimePassU = TarIter.CalTimeatU(m_TarArgLat)
        return OUTINFO(Tend=m_TimePassU, conArgLat=out.ConArgLat, tarArgLat=out.TarArgLat)

    def FindDecTime(self, TarSat, ConSat, Num:int):
        # 计算两个卫星在指定轨道周期后，主目标到达交线的时间
        TarIni = TarSat.GetIniOrb()
        ConIni = ConSat.GetIniOrb()
        dT = (TarIni.GetEpoch() - ConIni.GetEpoch()).sec
        if dT > 0:
            T0 = TarIni.GetEpoch() + TimeDelta(Num*TarIni.GetTp(), format='sec')
        else:
            T0 = ConIni.GetEpoch() + TimeDelta(Num*TarIni.GetTp(), format='sec')
        TarIter = TarSat.Propagate(T0)
        ConIter = ConSat.Propagate(T0)
        out = self.CalMaxAngle(TarIter, ConIter)
        m_TerArgLat = out.TarArgLat
        m_TimePassU = TarIter.CalTimeatU(m_TerArgLat)
        return OUTINFO(Tend=m_TimePassU, conArgLat=out.ConArgLat, tarArgLat=out.TarArgLat)

    def CubicSplining(self,p1,p2,p3,p4,t1,t2):
        c0 = p1
        DET = t1**3 * t2**2 + t1**2*t2 + t1*t2**3 - t1**3*t2 - t1**2*t2**3 - t1*t2**2
        c1 = ((t2**3 - t2**2)*(p2-p1)+(t1**2-t1**3)*(p3-p1)+(t1**3*t2**2 - t1**2*t2**3)*(p4-p1))/DET
        c2 = ((t2 - t2**3)*(p2-p1)+(t1**3-t1)*(p3-p1)+(t1*t2**3 - t1**3*t2)*(p4-p1))/DET
        c3 = ((t2**2 - t2)*(p2-p1)+(t1 - t1**2)*(p3-p1)+(t1**2*t2 - t1*t2**2)*(p4-p1))/DET
        P = c2/c3; Q=c1/c3; R=c0/c3
        a = (3*Q-P*P)/3.0
        b = (2*P**3 -9*P*Q +27*R)/27.0
        delt = a**3/27.0 + b**2/4.0
        if delt > 0:
            bb1 = -b/2.0 + np.sqrt(delt)
            bb2 = -b/2.0 - np.sqrt(delt)
            out = np.cbrt(bb1)+np.cbrt(bb2)
            return out
        E0 = 2*np.sqrt(-a/3.0)
        cosphi = -b/2.0 / np.sqrt(-a**3/27.0)
        cosphi = np.clip(cosphi, -1, 1)
        phi = np.arccos(cosphi)
        Z1 = E0*np.cos(phi/3.0)
        Z2 = E0*np.cos(phi/3.0 + 2*np.pi/3.0)
        Z3 = E0*np.cos(phi/3.0 + 4*np.pi/3.0)
        out3 = Z3 - P/3.0
        return out3
    
