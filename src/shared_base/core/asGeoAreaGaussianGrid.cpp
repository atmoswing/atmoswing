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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include "asGeoAreaGaussianGrid.h"

asGeoAreaGaussianGrid::asGeoAreaGaussianGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, GaussianGridType type, float Level, float Height, int flatAllowed)
:
asGeoArea(CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    switch (type)
    {
    case (T62):
        {
            m_fullAxisY.resize(94);
            m_fullAxisY << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_fullAxisX = Array1DDouble::LinSpaced(577, -360, 720);
        }
    default:
        {
            asLogWarning("The Gaussian grid type was not correctly defined. T62 taken as default.");
            m_fullAxisY.resize(94);
            m_fullAxisY << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_fullAxisX = Array1DDouble::LinSpaced(577, -360, 720);
        }
    }

    if(!IsOnGrid(CornerUL)) asThrowException(_("The given area does not match a gaussian grid."));
    if(!IsOnGrid(CornerUR)) asThrowException(_("The given area does not match a gaussian grid."));
    if(!IsOnGrid(CornerLL)) asThrowException(_("The given area does not match a gaussian grid."));
    if(!IsOnGrid(CornerLR)) asThrowException(_("The given area does not match a gaussian grid."));
}

asGeoAreaGaussianGrid::asGeoAreaGaussianGrid(double Xmin, int Xptsnb, double Ymin, int Yptsnb, GaussianGridType type, float Level, float Height, int flatAllowed)
:
asGeoArea(Level, Height)
{
    switch (type)
    {
    case (T62):
        {
            m_fullAxisY.resize(94);
            m_fullAxisY << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_fullAxisX = Array1DDouble::LinSpaced(577, -360, 720);
        }
    default:
        {
            asLogWarning("The Gaussian grid type was not correctly defined. T62 taken as default.");
            m_fullAxisY.resize(94);
            m_fullAxisY << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_fullAxisX = Array1DDouble::LinSpaced(577, -360, 720);
        }
    }

    // Check input
    if(!IsOnGrid(Xmin, Ymin)) asThrowException(_("The given area does not match a gaussian grid."));

    // Get real size to generate parent member variables
    int indexXmin = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size()-1], Xmin, 0.01);
    int indexYmin = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size()-1], Ymin, 0.01);
    wxASSERT(indexXmin>=0);
    wxASSERT(indexYmin>=0);
    if (m_fullAxisX.size()<=indexXmin+Xptsnb-1) asThrowException(_("The given width exceeds the grid size of the guassian grid."));
    if (m_fullAxisY.size()<=indexYmin+Yptsnb-1) asThrowException(_("The given height exceeds the grid size of the guassian grid."));
    double Xwidth = m_fullAxisX[indexXmin+Xptsnb-1] - m_fullAxisX[indexXmin];
    double Ywidth = m_fullAxisY[indexYmin+Yptsnb-1] - m_fullAxisY[indexYmin];

    // Regenerate with correct sizes
    Generate(Xmin, Xwidth, Ymin, Ywidth, flatAllowed);
}

asGeoAreaGaussianGrid::~asGeoAreaGaussianGrid()
{
    //dtor
}

int asGeoAreaGaussianGrid::GetXaxisPtsnb()
{
    double Xmin = GetXmin();
    int XminIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size()-1], Xmin, 0.01);
    double Xmax = GetXmax();
    int XmaxIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size()-1], Xmax, 0.01);

    // Get axis size
    return abs(XmaxIndex-XminIndex)+1;
}

int asGeoAreaGaussianGrid::GetYaxisPtsnb()
{
    double Ymin = GetYmin();
    int YminIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size()-1], Ymin, 0.01);
    double Ymax = GetYmax();
    int YmaxIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size()-1], Ymax, 0.01);

    // Get axis size
    return abs(YmaxIndex-YminIndex)+1;
}

Array1DDouble asGeoAreaGaussianGrid::GetXaxis()
{
    double Xmin = GetXmin();
    int XminIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size()-1], Xmin, 0.01);
    double Xmax = GetXmax();
    int XmaxIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size()-1], Xmax, 0.01);

    return m_fullAxisX.segment(XminIndex, XmaxIndex-XminIndex+1);
}

Array1DDouble asGeoAreaGaussianGrid::GetYaxis()
{
    double Ymin = GetYmin();
    int YminIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size()-1], Ymin, 0.01);
    double Ymax = GetYmax();
    int YmaxIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size()-1], Ymax, 0.01);

    return m_fullAxisY.segment(YminIndex, YmaxIndex-YminIndex+1);
}

bool asGeoAreaGaussianGrid::IsOnGrid(const Coo &point)
{
    if (!IsRectangle()) return false;

    int foundU = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size()-1], point.x, 0.01);
    if ( (foundU==asNOT_FOUND) || (foundU==asOUT_OF_RANGE) ) return false;

    int foundV = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size()-1], point.y, 0.01);
    if ( (foundV==asNOT_FOUND) || (foundV==asOUT_OF_RANGE) ) return false;

    return true;
}

bool asGeoAreaGaussianGrid::IsOnGrid(double Xcoord, double Ycoord)
{
    int foundU = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size()-1], Xcoord, 0.01);
    if ( (foundU==asNOT_FOUND) || (foundU==asOUT_OF_RANGE) ) return false;

    int foundV = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size()-1], Ycoord, 0.01);
    if ( (foundV==asNOT_FOUND) || (foundV==asOUT_OF_RANGE) ) return false;

    return true;
}
