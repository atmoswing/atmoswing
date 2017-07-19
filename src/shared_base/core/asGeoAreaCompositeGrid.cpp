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

#include "asGeoAreaCompositeGrid.h"
#include "asGeoAreaCompositeRegularGrid.h"
#include "asGeoAreaGaussianGrid.h"
#include "asGeoAreaCompositeGaussianGrid.h"

asGeoAreaCompositeGrid *asGeoAreaCompositeGrid::GetInstance(const wxString &type, double Xmin, int Xptsnb, double Xstep,
                                                            double Ymin, int Yptsnb, double Ystep, float Level,
                                                            float Height, int flatAllowed)
{
    // If empty, set Regular.
    if (type.IsEmpty()) {
        wxLogVerbose(_("The given grid type is empty. A regular grid has been considered."));
        double Xwidth = (double) (Xptsnb - 1) * Xstep;
        double Ywidth = (double) (Yptsnb - 1) * Ystep;
        asGeoAreaCompositeGrid *area = new asGeoAreaCompositeRegularGrid(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep,
                                                                         Level, Height, flatAllowed);
        return area;
    } else if (type.IsSameAs("Regular", false)) {
        double Xwidth = (double) (Xptsnb - 1) * Xstep;
        double Ywidth = (double) (Yptsnb - 1) * Ystep;
        asGeoAreaCompositeGrid *area = new asGeoAreaCompositeRegularGrid(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep,
                                                                         Level, Height, flatAllowed);
        return area;
    } else if (type.IsSameAs("GaussianT62", false)) {
        asGeoAreaCompositeGrid *area = new asGeoAreaCompositeGaussianGrid(Xmin, Xptsnb, Ymin, Yptsnb,
                                                                          asGeo::GaussianT62, Level, Height,
                                                                          flatAllowed);
        return area;
    } else if (type.IsSameAs("GaussianT382", false)) {
        asGeoAreaCompositeGrid *area = new asGeoAreaCompositeGaussianGrid(Xmin, Xptsnb, Ymin, Yptsnb,
                                                                          asGeo::GaussianT382, Level, Height,
                                                                          flatAllowed);
        return area;
    } else {
        wxLogError(_("Given grid type: %s"), type);
        asThrowException("The given grid type doesn't correspond to any existing option.");
    }
}

a1d asGeoAreaCompositeGrid::GetXaxis(const wxString &type, double Xmin, double Xmax, double Xstep)
{
    a1d axis;

    if (type.IsSameAs("Regular", false)) {
        wxASSERT(Xstep > 0);
        int ni = (int) asTools::Round(360 / Xstep);
        axis = a1d::LinSpaced(ni * 3 + 1, -360, 720);
    } else if (type.IsSameAs("GaussianT62", false)) {
        asGeoAreaGaussianGrid::BuildLonAxis(axis, asGeo::GaussianT62);
    } else if (type.IsSameAs("GaussianT382", false)) {
        asGeoAreaGaussianGrid::BuildLonAxis(axis, asGeo::GaussianT382);
    } else {
        wxLogError(_("Cannot build axis for the given grid type (%s)."), type);
        asThrowException(wxString::Format(_("Cannot build axis for the given grid type (%s)."), type));
    }

    wxASSERT(axis.size() > 0);

    int start = asTools::SortedArraySearchClosest(&axis[0], &axis[axis.size() - 1], Xmin);
    int end = asTools::SortedArraySearchClosest(&axis[0], &axis[axis.size() - 1], Xmax);

    wxASSERT(start >= 0);
    wxASSERT(end >= 0);
    wxASSERT(end >= start);
    wxASSERT(axis.size() > end - start + 1);

    return axis.segment(start, end - start + 1);
}

a1d asGeoAreaCompositeGrid::GetYaxis(const wxString &type, double Ymin, double Ymax, double Ystep)
{
    a1d axis;

    if (type.IsSameAs("Regular", false)) {
        wxASSERT(Ystep > 0);
        int ni = (int) asTools::Round(180 / Ystep);
        axis = a1d::LinSpaced(ni + 1, -90, 90);
    } else if (type.IsSameAs("GaussianT62", false)) {
        asGeoAreaGaussianGrid::BuildLatAxis(axis, asGeo::GaussianT62);
    } else if (type.IsSameAs("GaussianT382", false)) {
        asGeoAreaGaussianGrid::BuildLatAxis(axis, asGeo::GaussianT382);
    } else {
        wxLogError(_("Cannot build axis for the given grid type (%s)."), type);
        asThrowException(wxString::Format(_("Cannot build axis for the given grid type (%s)."), type));
    }

    wxASSERT(axis.size() > 0);

    int start = asTools::SortedArraySearchClosest(&axis[0], &axis[axis.size() - 1], Ymin);
    int end = asTools::SortedArraySearchClosest(&axis[0], &axis[axis.size() - 1], Ymax);

    wxASSERT(start >= 0);
    wxASSERT(end >= 0);
    wxASSERT(end >= start);
    wxASSERT(axis.size() > end - start + 1);

    return axis.segment(start, end - start + 1);
}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL,
                                               const Coo &CornerLR, float Level, float Height, int flatAllowed)
        : asGeoAreaComposite(CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    m_gridType = Regular;
}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level,
                                               float Height, int flatAllowed)
        : asGeoAreaComposite(Xmin, Xwidth, Ymin, Ywidth, Level, Height, flatAllowed)
{
    m_gridType = Regular;
}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(float Level, float Height)
        : asGeoAreaComposite(Level, Height)
{
    m_gridType = Regular;
}

int asGeoAreaCompositeGrid::GetXaxisPtsnb()
{
    int ptsLon = 0;

    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        if (iArea == 0) {
            ptsLon += GetXaxisCompositePtsnb(iArea);
        } else if (iArea == 4) {
            // Do nothing here
        } else {
            if (GetComposite(iArea).GetYmin() == GetComposite(iArea - 1).GetYmin()) {
                if (GetXaxisCompositeEnd(iArea) == m_axisXmax) {
                    ptsLon += GetXaxisCompositePtsnb(iArea) - 1;
                } else {
                    ptsLon += GetXaxisCompositePtsnb(iArea);
                }
            }
        }
    }

    return ptsLon;
}

int asGeoAreaCompositeGrid::GetYaxisPtsnb()
{
    int ptsLat = 0;

    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        if (iArea == 0) {
            ptsLat += GetYaxisCompositePtsnb(iArea);
        } else if (iArea == 4) {
            // Do nothing here
        } else {
            if (GetComposite(iArea).GetXmin() == GetComposite(iArea - 1).GetXmin()) {
                if (GetYaxisCompositeEnd(iArea) == m_axisYmax) {
                    ptsLat += GetYaxisCompositePtsnb(iArea) - 1;
                } else {
                    ptsLat += GetYaxisCompositePtsnb(iArea);
                }
            }
        }
    }

    return ptsLat;
}

double asGeoAreaCompositeGrid::GetXaxisWidth() const
{
    double widthLon = 0;

    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        if (iArea == 0) {
            widthLon += GetXaxisCompositeWidth(iArea);
        } else if (iArea == 4) {
            // Do nothing here
        } else {
            if (GetComposite(iArea).GetYmin() == GetComposite(iArea - 1).GetYmin()) {
                widthLon += GetXaxisCompositeWidth(iArea);
            }
        }
    }

    return widthLon;
}

double asGeoAreaCompositeGrid::GetYaxisWidth() const
{
    double widthLat = 0;

    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        if (iArea == 0) {
            widthLat += GetYaxisCompositeWidth(iArea);
        } else if (iArea == 4) {
            // Do nothing here
        } else {
            if (GetComposite(iArea).GetXmin() == GetComposite(iArea - 1).GetXmin()) {
                widthLat += GetYaxisCompositeWidth(iArea);
            }
        }
    }

    return widthLat;
}

a1d asGeoAreaCompositeGrid::GetXaxis()
{
    a1d Xaxis;

    wxASSERT(GetNbComposites() > 0);

    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        if (iArea == 0) {
            Xaxis = GetXaxisComposite(iArea);
        } else if (iArea == 4) {
            // Do nothing here
        } else {
            if (GetComposite(iArea).GetYmin() == GetComposite(iArea - 1).GetYmin()) {
                a1d Xaxisbis = GetXaxisComposite(iArea);

                if (GetXaxisCompositeEnd(iArea) == m_axisXmax) {
                    a1d Xaxisfinal(Xaxisbis.size() + Xaxis.size() - 1);
                    Xaxisfinal.head(Xaxisbis.size()) = Xaxisbis;
                    for (int i = 1; i < Xaxis.size(); i++) {
                        Xaxisfinal[Xaxisbis.size() - 1 + i] = Xaxis[i] + m_axisXmax;
                    }
                    return Xaxisfinal;
                } else {
                    a1d Xaxisfinal(Xaxisbis.size() + Xaxis.size());
                    Xaxisfinal.head(Xaxisbis.size()) = Xaxisbis;
                    for (int i = 0; i < Xaxis.size(); i++) {
                        Xaxisfinal[Xaxisbis.size() + i] = Xaxis[i] + m_axisXmax;
                    }
                    return Xaxisfinal;
                }
            }
        }
    }

    return Xaxis;
}

a1d asGeoAreaCompositeGrid::GetYaxis()
{
    a1d Yaxis;

    wxASSERT(GetNbComposites() > 0);

    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        if (iArea == 0) {
            Yaxis = GetYaxisComposite(iArea);
        } else if (iArea == 4) {
            // Do nothing here
        } else {
            if (GetComposite(iArea).GetXmin() == GetComposite(iArea - 1).GetXmin()) {
                wxLogError(_("This function has not been tested"));

                a1d Yaxisbis = GetYaxisComposite(iArea);

                if (GetXaxisCompositeEnd(iArea) == m_axisXmax) {
                    a1d Yaxisfinal(Yaxisbis.size() + Yaxis.size() - 1);
                    Yaxisfinal.head(Yaxisbis.size()) = Yaxisbis;
                    for (int i = 1; i < Yaxis.size(); i++) {
                        Yaxisfinal[Yaxisbis.size() - 1 + i] = Yaxis[i] + m_axisYmax;
                    }
                    return Yaxisfinal;
                } else {
                    a1d Yaxisfinal(Yaxisbis.size() + Yaxis.size());
                    Yaxisfinal.head(Yaxisbis.size()) = Yaxisbis;
                    for (int i = 0; i < Yaxis.size(); i++) {
                        Yaxisfinal[Yaxisbis.size() + i] = Yaxis[i] + m_axisYmax;
                    }
                    return Yaxisfinal;
                }
            }
        }
    }

    return Yaxis;
}

void asGeoAreaCompositeGrid::SetLastRowAsNewComposite()
{
    wxASSERT(GetNbComposites() == 1);
    asGeoArea area1 = GetComposite(0);
    Coo cornerLR1 = area1.GetCornerLR();
    Coo cornerUR1 = area1.GetCornerUR();

    a1d xAxis = GetXaxisComposite(0);
    cornerLR1.x = xAxis[xAxis.size() - 2];
    cornerUR1.x = xAxis[xAxis.size() - 2];

    area1.SetCornerLR(cornerLR1);
    area1.SetCornerUR(cornerUR1);

    Coo cornerLR2 = area1.GetCornerLR();
    Coo cornerUR2 = area1.GetCornerUR();
    Coo cornerLL2 = area1.GetCornerLL();
    Coo cornerUL2 = area1.GetCornerUL();
    cornerLR2.x = 0;
    cornerUR2.x = 0;
    cornerLL2.x = 0;
    cornerUL2.x = 0;

    asGeoArea area2(cornerUL2, cornerUR2, cornerLL2, cornerLR2, m_level, m_height, asFLAT_ALLOWED);

    // Add the composite in this specific order to be consistent with the other implementations.
    m_composites.clear();
    m_composites.push_back(area2);
    m_composites.push_back(area1);
}

void asGeoAreaCompositeGrid::RemoveLastRowOnComposite(int i)
{
    wxASSERT(GetNbComposites() > 1);
    asGeoArea area = GetComposite(i);
    Coo cornerLR = area.GetCornerLR();
    Coo cornerUR = area.GetCornerUR();

    a1d xAxis = GetXaxisComposite(i);
    cornerLR.x = xAxis[xAxis.size() - 2];
    cornerUR.x = xAxis[xAxis.size() - 2];

    area.SetCornerLR(cornerLR);
    area.SetCornerUR(cornerUR);

    m_composites[i] = area;
}
