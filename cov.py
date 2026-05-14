import numpy as np
from sgp4.api import Satrec, WGS84

def get_state_vector(tle_line1, tle_line2, jd, fr):
    """
    解析TLE并返回TEME坐标系中的位置与速度.
    Parses TLE and returns the position and velocity in the TEME frame.
    """
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    e, r, v = satellite.sgp4(jd, fr)
    
    if e != 0:
        raise ValueError(f"SGP4 Error Code: {e}")
        
    return np.array(r), np.array(v)

def calculate_rtn_to_eci_rotation(r, v):
    """
    根据位置（r）和速度（v）计算从RTN（径向、横向、法向）坐标系到ECI坐标系的旋转矩阵.
    Calculates the rotation matrix from the RTN (Radial, Transverse, Normal) 
    frame to the ECI (TEME) frame based on position (r) and velocity (v).
    """
    # Radial unit vector (direction of the satellite from Earth center)
    u_R = r / np.linalg.norm(r)
    
    # Normal unit vector (perpendicular to the orbital plane)
    h = np.cross(r, v) # Angular momentum vector
    u_N = h / np.linalg.norm(h)
    
    # Transverse unit vector (In-track, completes the right-handed system)
    u_T = np.cross(u_N, u_R)
    
    # Rotation matrix from RTN to ECI
    # R_matrix = [u_R, u_T, u_N] (columns)
    R = np.column_stack((u_R, u_T, u_N))
    return R

def generate_covariance_matrix(tle_line1, tle_line2, jd, fr, rtn_errors_km):
    """
    在TEME（地心惯性系）框架下生成3x3的位置协方差矩阵.
    Generates a 3x3 position covariance matrix in the TEME (ECI) frame.
    """
    # 1. Get position and velocity from TLE
    r, v = get_state_vector(tle_line1, tle_line2, jd, fr)
    
    # 2. Define the empirical covariance matrix in the RTN frame
    # Assuming the errors are standard deviations (sigma) in km
    # Variance = sigma^2. Assuming no cross-correlation for this basic model.
    sigma_r, sigma_t, sigma_n = rtn_errors_km
    
    P_RTN = np.array([
        [sigma_r**2, 0,          0],
        [0,          sigma_t**2, 0],
        [0,          0,          sigma_n**2]
    ])
    
    # 3. Get the rotation matrix
    R = calculate_rtn_to_eci_rotation(r, v)
    
    # 4. Rotate the covariance matrix into the ECI frame: P_ECI = R * P_RTN * R^T
    P_ECI = R @ P_RTN @ R.T
    
    return r, v, P_ECI

# --- Example Usage ---
if __name__ == "__main__":
    # Example TLE for the ISS
    line1 = "1 25544U 98067A   23282.52918804  .00015568  00000+0  28258-3 0  9997"
    line2 = "2 25544  51.6416  67.6534 0005230 293.7547 165.7486 15.49842407419614"
    
    # Time of interest: Julian Date and Fractional part of the day
    # Let's use the epoch time of the TLE roughly for this example
    jd = 2460226.5
    fr = 0.02918804 
    
    # Typical TLE 1-sigma errors in kilometers: [Radial, Transverse(In-track), Normal(Cross-track)]
    # Note: In-track error is typically the largest due to atmospheric drag uncertainties.
    empirical_errors = [1.0, 5.0, 1.0] 
    
    r, v, covariance_matrix = generate_covariance_matrix(line1, line2, jd, fr, empirical_errors)
    
    print("Position Vector (km):")
    print(r)
    print("\nEmpirical Position Covariance Matrix in ECI frame (km^2):")
    np.set_printoptions(precision=4, suppress=True)
    print(covariance_matrix)