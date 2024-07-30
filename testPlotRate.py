import numpy as np

if __name__ == "__main__":
    m_list = []
    m = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    for i in range(3):
        if m[i] > 0.5:
            m_list.append(1 * (m - m[i] > 0))
        else:
            m_list.append(1 * (m - m[i] >= 0))
    print(m_list)
    m_list = m_list + m_list
    print(m_list)


