import numpy as np

def calculate_heading_direction(acc_x, acc_y, acc_z, mag_x, mag_y, mag_z, gyr_x, gyr_y, gyr_z):
    phi = np.arctan2(acc_y, acc_z)
    theta = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))
    M_y = mag_x * np.sin(phi) * np.sin(theta) + mag_y * np.cos(phi) - mag_z * np.sin(phi) * np.cos(theta)
    M_x = mag_x * np.cos(theta) + mag_z * np.sin(theta)
    psi = np.arctan2(M_y, M_x)

    phi_star = gyr_x + gyr_y * np.sin(phi) * np.tan(theta) + gyr_z * np.cos(phi) * np.tan(theta)
    theta_star = gyr_y * np.cos(phi) - gyr_z * np.sin(phi)
    psi_star = gyr_y * (np.sin(phi) / np.cos(theta)) + gyr_z * (np.cos(phi) / np.cos(theta))

    G_Q = np.array([
        [np.cos(phi_star/2)*np.cos(theta_star/2)*np.cos(psi_star/2) + np.sin(phi_star/2)*np.sin(theta_star/2)*np.sin(psi_star/2)],
        [np.sin(phi_star/2)*np.cos(theta_star/2)*np.cos(psi_star/2) - np.cos(phi_star/2)*np.sin(theta_star/2)*np.sin(psi_star/2)],
        [np.cos(phi_star/2)*np.sin(theta_star/2)*np.cos(psi_star/2) + np.sin(phi_star/2)*np.cos(theta_star/2)*np.sin(psi_star/2)],
        [np.cos(phi_star/2)*np.cos(theta_star/2)*np.sin(psi_star/2) - np.sin(phi_star/2)*np.sin(theta_star/2)*np.cos(psi_star/2)]
    ])

    R = 0.5 * np.array([
        [0, -gyr_x, -gyr_y],
        [gyr_x, 0, -gyr_z],
        [gyr_y, gyr_z, 0]
    ]).dot(G_Q)
    G_x = 2 * (R[0, 0]*R[0, 2] + R[0, 1]*R[1, 2])
    G_y = 1 - 2 * (R[1, 2]**2 + R[0, 2]**2)
    heading_gyro = np.arctan2(G_x, G_y)
    heading_magnetometer = np.arctan2(M_y, M_x)
    final_heading_direction = np.degrees(np.mean([heading_gyro, heading_magnetometer]))

    return final_heading_direction