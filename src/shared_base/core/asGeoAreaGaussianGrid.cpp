/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asGeoAreaGaussianGrid.h"

asGeoAreaGaussianGrid::asGeoAreaGaussianGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, GaussianGridType type, float Level, float Height, int flatAllowed)
:
asGeoArea(coosys, CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    switch (type)
    {
    case (T62):
        {
            m_FullAxisV.resize(94);
            m_FullAxisV << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_FullAxisU = Array1DDouble::LinSpaced(577, -360, 720);
        }
    default:
        {
            asLogWarning("The Gaussian grid type was not correctly defined. T62 taken as default.");
            m_FullAxisV.resize(94);
            m_FullAxisV << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_FullAxisU = Array1DDouble::LinSpaced(577, -360, 720);
        }
    }

    if(!IsOnGrid(CornerUL)) asThrowException(_("The given area does not match a gaussian grid."));
    if(!IsOnGrid(CornerUR)) asThrowException(_("The given area does not match a gaussian grid."));
    if(!IsOnGrid(CornerLL)) asThrowException(_("The given area does not match a gaussian grid."));
    if(!IsOnGrid(CornerLR)) asThrowException(_("The given area does not match a gaussian grid."));
}

asGeoAreaGaussianGrid::asGeoAreaGaussianGrid(CoordSys coosys, double Umin, int Uptsnb, double Vmin, int Vptsnb, GaussianGridType type, float Level, float Height, int flatAllowed)
:
asGeoArea(coosys, Level, Height)
{
    switch (type)
    {
    case (T62):
        {
            m_FullAxisV.resize(94);
            m_FullAxisV << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_FullAxisU = Array1DDouble::LinSpaced(577, -360, 720);
        }
    default:
        {
            asLogWarning("The Gaussian grid type was not correctly defined. T62 taken as default.");
            m_FullAxisV.resize(94);
            m_FullAxisV << -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762,  06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
            m_FullAxisU = Array1DDouble::LinSpaced(577, -360, 720);
        }
    }

    // Check input
    if(!IsOnGrid(Umin, Vmin)) asThrowException(_("The given area does not match a gaussian grid."));

    // Get real size to generate parent member variables
    int indexUmin = asTools::SortedArraySearch(&m_FullAxisU[0], &m_FullAxisU[m_FullAxisU.size()-1], Umin, 0.01);
    int indexVmin = asTools::SortedArraySearch(&m_FullAxisV[0], &m_FullAxisV[m_FullAxisV.size()-1], Vmin, 0.01);
    wxASSERT(indexUmin>=0);
    wxASSERT(indexVmin>=0);
    if (m_FullAxisU.size()<=indexUmin+Uptsnb-1) asThrowException(_("The given width exceeds the grid size of the guassian grid."));
    if (m_FullAxisV.size()<=indexVmin+Vptsnb-1) asThrowException(_("The given height exceeds the grid size of the guassian grid."));
    double Uwidth = m_FullAxisU[indexUmin+Uptsnb-1] - m_FullAxisU[indexUmin];
    double Vwidth = m_FullAxisV[indexVmin+Vptsnb-1] - m_FullAxisV[indexVmin];

    // Regenerate with correct sizes
    Generate(Umin, Uwidth, Vmin, Vwidth, flatAllowed);
}

asGeoAreaGaussianGrid::~asGeoAreaGaussianGrid()
{
    //dtor
}

int asGeoAreaGaussianGrid::GetUaxisPtsnb()
{
    double umin = GetUmin();
    int UminIndex = asTools::SortedArraySearch(&m_FullAxisU[0], &m_FullAxisU[m_FullAxisU.size()-1], umin, 0.01);
    double umax = GetUmax();
    int UmaxIndex = asTools::SortedArraySearch(&m_FullAxisU[0], &m_FullAxisU[m_FullAxisU.size()-1], umax, 0.01);

    // Get axis size
    return abs(UmaxIndex-UminIndex)+1;
}

int asGeoAreaGaussianGrid::GetVaxisPtsnb()
{
    double vmin = GetVmin();
    int VminIndex = asTools::SortedArraySearch(&m_FullAxisV[0], &m_FullAxisV[m_FullAxisV.size()-1], vmin, 0.01);
    double vmax = GetVmax();
    int VmaxIndex = asTools::SortedArraySearch(&m_FullAxisV[0], &m_FullAxisV[m_FullAxisV.size()-1], vmax, 0.01);

    // Get axis size
    return abs(VmaxIndex-VminIndex)+1;
}

Array1DDouble asGeoAreaGaussianGrid::GetUaxis()
{
    double umin = GetUmin();
    int UminIndex = asTools::SortedArraySearch(&m_FullAxisU[0], &m_FullAxisU[m_FullAxisU.size()-1], umin, 0.01);
    double umax = GetUmax();
    int UmaxIndex = asTools::SortedArraySearch(&m_FullAxisU[0], &m_FullAxisU[m_FullAxisU.size()-1], umax, 0.01);

    return m_FullAxisU.segment(UminIndex, UmaxIndex-UminIndex+1);
}

Array1DDouble asGeoAreaGaussianGrid::GetVaxis()
{
    double vmin = GetVmin();
    int VminIndex = asTools::SortedArraySearch(&m_FullAxisV[0], &m_FullAxisV[m_FullAxisV.size()-1], vmin, 0.01);
    double vmax = GetVmax();
    int VmaxIndex = asTools::SortedArraySearch(&m_FullAxisV[0], &m_FullAxisV[m_FullAxisV.size()-1], vmax, 0.01);

    return m_FullAxisV.segment(VminIndex, VmaxIndex-VminIndex+1);
}

bool asGeoAreaGaussianGrid::IsOnGrid(const Coo &point)
{
    if (!IsRectangle()) return false;

    int foundU = asTools::SortedArraySearch(&m_FullAxisU[0], &m_FullAxisU[m_FullAxisU.size()-1], point.u, 0.01);
    if ( (foundU==asNOT_FOUND) || (foundU==asOUT_OF_RANGE) ) return false;

    int foundV = asTools::SortedArraySearch(&m_FullAxisV[0], &m_FullAxisV[m_FullAxisV.size()-1], point.v, 0.01);
    if ( (foundV==asNOT_FOUND) || (foundV==asOUT_OF_RANGE) ) return false;

    return true;
}

bool asGeoAreaGaussianGrid::IsOnGrid(double Ucoord, double Vcoord)
{
    int foundU = asTools::SortedArraySearch(&m_FullAxisU[0], &m_FullAxisU[m_FullAxisU.size()-1], Ucoord, 0.01);
    if ( (foundU==asNOT_FOUND) || (foundU==asOUT_OF_RANGE) ) return false;

    int foundV = asTools::SortedArraySearch(&m_FullAxisV[0], &m_FullAxisV[m_FullAxisV.size()-1], Vcoord, 0.01);
    if ( (foundV==asNOT_FOUND) || (foundV==asOUT_OF_RANGE) ) return false;

    return true;
}
