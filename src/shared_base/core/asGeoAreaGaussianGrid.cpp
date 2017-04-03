/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asGeoAreaGaussianGrid.h"

asGeoAreaGaussianGrid::asGeoAreaGaussianGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL,
                                             const Coo &CornerLR, asGeo::GridType type, float Level, float Height,
                                             int flatAllowed)
        : asGeoArea(CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    m_gridType = type;

    BuildLonAxis(m_fullAxisX, type);
    BuildLatAxis(m_fullAxisY, type);

    if (!IsOnGrid(CornerUL))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(CornerUR))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(CornerLL))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(CornerLR))
        asThrowException(_("The given area does not match a gaussian grid."));
}

asGeoAreaGaussianGrid::asGeoAreaGaussianGrid(double Xmin, int Xptsnb, double Ymin, int Yptsnb, asGeo::GridType type,
                                             float Level, float Height, int flatAllowed)
        : asGeoArea(Level, Height)
{
    m_gridType = type;

    BuildLonAxis(m_fullAxisX, type);
    BuildLatAxis(m_fullAxisY, type);

    // Check input
    if (!IsOnGrid(Xmin, Ymin))
        asThrowException(_("The given area does not match a gaussian grid."));

    // Get real size to generate parent member variables
    int indexXmin = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmin, 0.01);
    int indexYmin = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymin, 0.01);
    wxASSERT(indexXmin >= 0);
    wxASSERT(indexYmin >= 0);
    if (m_fullAxisX.size() <= indexXmin + Xptsnb - 1)
        asThrowException(_("The given width exceeds the grid size of the guassian grid."));
    if (m_fullAxisY.size() <= indexYmin + Yptsnb - 1)
        asThrowException(_("The given height exceeds the grid size of the guassian grid."));
    double Xwidth = m_fullAxisX[indexXmin + Xptsnb - 1] - m_fullAxisX[indexXmin];
    double Ywidth = m_fullAxisY[indexYmin + Yptsnb - 1] - m_fullAxisY[indexYmin];

    // Regenerate with correct sizes
    Generate(Xmin, Xwidth, Ymin, Ywidth, flatAllowed);
}

asGeoAreaGaussianGrid::~asGeoAreaGaussianGrid()
{
    //dtor
}

void asGeoAreaGaussianGrid::BuildLonAxis(Array1DDouble &axis, const asGeo::GridType &type)
{
    int ni = 0;
    switch (type) {
        case (GaussianT62): {
            ni = 192;
            break;
        }
        case (GaussianT382): {
            ni = 1152;
            break;
        }
        default: {
            asThrowException("The Gaussian grid type was not correctly defined.");
        }
    }

    axis = Array1DDouble::LinSpaced(ni * 3 + 1, -360, 720);
}

void asGeoAreaGaussianGrid::BuildLatAxis(Array1DDouble &axis, const asGeo::GridType &type)
{
    switch (type) {
        case (GaussianT62): {
            axis.resize(94);
            axis << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522,
                    -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571,
                    -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619,
                    -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667,
                    -04.762, -02.857, -00.952, 00.952, 02.857, 04.762, 06.667, 08.571, 10.476, 12.381, 14.286, 16.190,
                    18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047,
                    40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903,
                    63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753,
                    86.653, 88.542;
            break;
        }
        case (GaussianT382): {
            axis.resize(576);
            axis << -89.761, -89.451, -89.14, -88.828, -88.516, -88.204, -87.892, -87.58, -87.268, -86.955, -86.643,
                    -86.331, -86.019, -85.707, -85.394, -85.082, -84.77, -84.458, -84.146, -83.833, -83.521, -83.209,
                    -82.897, -82.584, -82.272, -81.96, -81.648, -81.336, -81.023, -80.711, -80.399, -80.087, -79.774,
                    -79.462, -79.15, -78.838, -78.525, -78.213, -77.901, -77.589, -77.277, -76.964, -76.652, -76.34,
                    -76.028, -75.715, -75.403, -75.091, -74.779, -74.467, -74.154, -73.842, -73.53, -73.218, -72.905,
                    -72.593, -72.281, -71.969, -71.656, -71.344, -71.032, -70.72, -70.408, -70.095, -69.783, -69.471,
                    -69.159, -68.846, -68.534, -68.222, -67.91, -67.598, -67.285, -66.973, -66.661, -66.349, -66.036,
                    -65.724, -65.412, -65.1, -64.787, -64.475, -64.163, -63.851, -63.539, -63.226, -62.914, -62.602,
                    -62.29, -61.977, -61.665, -61.353, -61.041, -60.728, -60.416, -60.104, -59.792, -59.48, -59.167,
                    -58.855, -58.543, -58.231, -57.918, -57.606, -57.294, -56.982, -56.67, -56.357, -56.045, -55.733,
                    -55.421, -55.108, -54.796, -54.484, -54.172, -53.859, -53.547, -53.235, -52.923, -52.611, -52.298,
                    -51.986, -51.674, -51.362, -51.049, -50.737, -50.425, -50.113, -49.8, -49.488, -49.176, -48.864,
                    -48.552, -48.239, -47.927, -47.615, -47.303, -46.99, -46.678, -46.366, -46.054, -45.742, -45.429,
                    -45.117, -44.805, -44.493, -44.18, -43.868, -43.556, -43.244, -42.931, -42.619, -42.307, -41.995,
                    -41.683, -41.37, -41.058, -40.746, -40.434, -40.121, -39.809, -39.497, -39.185, -38.872, -38.56,
                    -38.248, -37.936, -37.624, -37.311, -36.999, -36.687, -36.375, -36.062, -35.75, -35.438, -35.126,
                    -34.814, -34.501, -34.189, -33.877, -33.565, -33.252, -32.94, -32.628, -32.316, -32.003, -31.691,
                    -31.379, -31.067, -30.755, -30.442, -30.13, -29.818, -29.506, -29.193, -28.881, -28.569, -28.257,
                    -27.944, -27.632, -27.32, -27.008, -26.696, -26.383, -26.071, -25.759, -25.447, -25.134, -24.822,
                    -24.51, -24.198, -23.886, -23.573, -23.261, -22.949, -22.637, -22.324, -22.012, -21.7, -21.388,
                    -21.075, -20.763, -20.451, -20.139, -19.827, -19.514, -19.202, -18.89, -18.578, -18.265, -17.953,
                    -17.641, -17.329, -17.016, -16.704, -16.392, -16.08, -15.768, -15.455, -15.143, -14.831, -14.519,
                    -14.206, -13.894, -13.582, -13.27, -12.957, -12.645, -12.333, -12.021, -11.709, -11.396, -11.084,
                    -10.772, -10.46, -10.147, -9.835, -9.523, -9.211, -8.899, -8.586, -8.274, -7.962, -7.65, -7.337,
                    -7.025, -6.713, -6.401, -6.088, -5.776, -5.464, -5.152, -4.84, -4.527, -4.215, -3.903, -3.591,
                    -3.278, -2.966, -2.654, -2.342, -2.029, -1.717, -1.405, -1.093, -0.781, -0.468, -0.156, 0.156,
                    0.468, 0.781, 1.093, 1.405, 1.717, 2.029, 2.342, 2.654, 2.966, 3.278, 3.591, 3.903, 4.215, 4.527,
                    4.84, 5.152, 5.464, 5.776, 6.088, 6.401, 6.713, 7.025, 7.337, 7.65, 7.962, 8.274, 8.586, 8.899,
                    9.211, 9.523, 9.835, 10.147, 10.46, 10.772, 11.084, 11.396, 11.709, 12.021, 12.333, 12.645, 12.957,
                    13.27, 13.582, 13.894, 14.206, 14.519, 14.831, 15.143, 15.455, 15.768, 16.08, 16.392, 16.704,
                    17.016, 17.329, 17.641, 17.953, 18.265, 18.578, 18.89, 19.202, 19.514, 19.827, 20.139, 20.451,
                    20.763, 21.075, 21.388, 21.7, 22.012, 22.324, 22.637, 22.949, 23.261, 23.573, 23.886, 24.198,
                    24.51, 24.822, 25.134, 25.447, 25.759, 26.071, 26.383, 26.696, 27.008, 27.32, 27.632, 27.944,
                    28.257,  28.569, 28.881, 29.193, 29.506, 29.818, 30.13, 30.442, 30.755, 31.067, 31.379, 31.691,
                    32.003, 32.316, 32.628, 32.94, 33.252, 33.565, 33.877, 34.189, 34.501, 34.814, 35.126, 35.438,
                    35.75, 36.062, 36.375, 36.687, 36.999, 37.311, 37.624, 37.936, 38.248, 38.56, 38.872, 39.185,
                    39.497, 39.809, 40.121, 40.434, 40.746, 41.058, 41.37, 41.683, 41.995, 42.307, 42.619, 42.931,
                    43.244, 43.556, 43.868, 44.18, 44.493, 44.805, 45.117, 45.429, 45.742, 46.054, 46.366, 46.678,
                    46.99, 47.303, 47.615, 47.927, 48.239, 48.552, 48.864, 49.176, 49.488, 49.8, 50.113, 50.425,
                    50.737, 51.049, 51.362, 51.674, 51.986, 52.298, 52.611, 52.923, 53.235, 53.547, 53.859, 54.172,
                    54.484, 54.796, 55.108, 55.421, 55.733, 56.045, 56.357, 56.67, 56.982, 57.294, 57.606, 57.918,
                    58.231, 58.543, 58.855, 59.167, 59.48, 59.792, 60.104, 60.416, 60.728, 61.041, 61.353, 61.665,
                    61.977, 62.29, 62.602, 62.914, 63.226, 63.539, 63.851, 64.163, 64.475, 64.787, 65.1, 65.412,
                    65.724, 66.036, 66.349, 66.661, 66.973, 67.285, 67.598, 67.91, 68.222, 68.534, 68.846, 69.159,
                    69.471, 69.783, 70.095, 70.408, 70.72, 71.032, 71.344, 71.656, 71.969, 72.281, 72.593, 72.905,
                    73.218, 73.53, 73.842, 74.154, 74.467, 74.779, 75.091, 75.403, 75.715, 76.028, 76.34, 76.652,
                    76.964, 77.277, 77.589, 77.901, 78.213, 78.525, 78.838, 79.15, 79.462, 79.774, 80.087, 80.399,
                    80.711, 81.023, 81.336, 81.648, 81.96, 82.272, 82.584, 82.897, 83.209, 83.521, 83.833, 84.146,
                    84.458, 84.77, 85.082, 85.394, 85.707, 86.019, 86.331, 86.643, 86.955, 87.268, 87.58, 87.892,
                    88.204, 88.516, 88.828, 89.14, 89.451, 89.761;
            break;
        }
        default: {
            asThrowException("The Gaussian grid type was not correctly defined.");
        }
    }
}

int asGeoAreaGaussianGrid::GetXaxisPtsnb() const
{
    double Xmin = GetXmin();
    int XminIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmin, 0.01);
    double Xmax = GetXmax();
    int XmaxIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmax, 0.01);

    // Get axis size
    return std::abs(XmaxIndex - XminIndex) + 1;
}

int asGeoAreaGaussianGrid::GetYaxisPtsnb() const
{
    double Ymin = GetYmin();
    int YminIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymin, 0.01);
    double Ymax = GetYmax();
    int YmaxIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymax, 0.01);

    // Get axis size
    return std::abs(YmaxIndex - YminIndex) + 1;
}

Array1DDouble asGeoAreaGaussianGrid::GetXaxis()
{
    double Xmin = GetXmin();
    int XminIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmin, 0.01);
    double Xmax = GetXmax();
    int XmaxIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmax, 0.01);

    return m_fullAxisX.segment(XminIndex, XmaxIndex - XminIndex + 1);
}

Array1DDouble asGeoAreaGaussianGrid::GetYaxis()
{
    double Ymin = GetYmin();
    int YminIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymin, 0.01);
    double Ymax = GetYmax();
    int YmaxIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymax, 0.01);

    return m_fullAxisY.segment(YminIndex, YmaxIndex - YminIndex + 1);
}

bool asGeoAreaGaussianGrid::IsOnGrid(const Coo &point) const
{
    if (!IsRectangle())
        return false;

    int foundU = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], point.x, 0.01);
    if ((foundU == asNOT_FOUND) || (foundU == asOUT_OF_RANGE))
        return false;

    int foundV = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], point.y, 0.01);
    if ((foundV == asNOT_FOUND) || (foundV == asOUT_OF_RANGE))
        return false;

    return true;
}

bool asGeoAreaGaussianGrid::IsOnGrid(double Xcoord, double Ycoord) const
{
    int foundU = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xcoord, 0.01);
    if ((foundU == asNOT_FOUND) || (foundU == asOUT_OF_RANGE))
        return false;

    int foundV = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ycoord, 0.01);
    if ((foundV == asNOT_FOUND) || (foundV == asOUT_OF_RANGE))
        return false;

    return true;
}
