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
#include "asTypeDefs.h"

asGeoAreaCompositeGrid *asGeoAreaCompositeGrid::GetInstance(const wxString &type, double Xmin, int Xptsnb, double Xstep,
                                                            double Ymin, int Yptsnb, double Ystep, float Level,
                                                            float Height, int flatAllowed)
{
    // If empty, set Regular.
    if (type.IsEmpty()) {
        asLogMessage(_("The given grid type is empty. A regular grid has been considered."));
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
        asGeoAreaGaussianGrid::GaussianGridType gaussianType = asGeoAreaGaussianGrid::T62;
        asGeoAreaCompositeGrid *area = new asGeoAreaCompositeGaussianGrid(Xmin, Xptsnb, Ymin, Yptsnb, gaussianType,
                                                                          Level, Height, flatAllowed);
        return area;
    } else {
        asLogError(wxString::Format(_("Given grid type: %s"), type));
        asThrowException("The given grid type doesn't correspond to any existing option.");
    }
}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL,
                                               const Coo &CornerLR, float Level, float Height, int flatAllowed)
        : asGeoAreaComposite(CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    m_gridType = Regular;
    m_lastRowInComposite = false;
}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level,
                                               float Height, int flatAllowed)
        : asGeoAreaComposite(Xmin, Xwidth, Ymin, Ywidth, Level, Height, flatAllowed)
{
    m_gridType = Regular;
    m_lastRowInComposite = false;
}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(float Level, float Height)
        : asGeoAreaComposite(Level, Height)
{
    m_gridType = Regular;
    m_lastRowInComposite = false;
}

int asGeoAreaCompositeGrid::GetXaxisPtsnb() const
{
    int ptsLon = 0;

    for (int i_area = 0; i_area < GetNbComposites(); i_area++) {
        if (i_area == 0) {
            ptsLon += GetXaxisCompositePtsnb(i_area);
        } else if (i_area == 4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetYmin() == GetComposite(i_area - 1).GetYmin()) {
                if (GetXaxisCompositeEnd(i_area) == m_axisXmax || m_lastRowInComposite) {
                    ptsLon += GetXaxisCompositePtsnb(i_area) - 1;
                } else {
                    ptsLon += GetXaxisCompositePtsnb(i_area);
                }
            }
        }
    }

    return ptsLon;
}

int asGeoAreaCompositeGrid::GetYaxisPtsnb() const
{
    int ptsLat = 0;

    for (int i_area = 0; i_area < GetNbComposites(); i_area++) {
        if (i_area == 0) {
            ptsLat += GetYaxisCompositePtsnb(i_area);
        } else if (i_area == 4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetXmin() == GetComposite(i_area - 1).GetXmin()) {
                if (GetYaxisCompositeEnd(i_area) == m_axisYmax) {
                    ptsLat += GetYaxisCompositePtsnb(i_area) - 1;
                } else {
                    ptsLat += GetYaxisCompositePtsnb(i_area);
                }
            }
        }
    }

    return ptsLat;
}

double asGeoAreaCompositeGrid::GetXaxisWidth() const
{
    double widthLon = 0;

    for (int i_area = 0; i_area < GetNbComposites(); i_area++) {
        if (i_area == 0) {
            widthLon += GetXaxisCompositeWidth(i_area);
        } else if (i_area == 4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetYmin() == GetComposite(i_area - 1).GetYmin()) {
                widthLon += GetXaxisCompositeWidth(i_area);
            }
        }
    }

    return widthLon;
}

double asGeoAreaCompositeGrid::GetYaxisWidth() const
{
    double widthLat = 0;

    for (int i_area = 0; i_area < GetNbComposites(); i_area++) {
        if (i_area == 0) {
            widthLat += GetYaxisCompositeWidth(i_area);
        } else if (i_area == 4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetXmin() == GetComposite(i_area - 1).GetXmin()) {
                widthLat += GetYaxisCompositeWidth(i_area);
            }
        }
    }

    return widthLat;
}

Array1DDouble asGeoAreaCompositeGrid::GetXaxis() const
{
    Array1DDouble Xaxis;

    wxASSERT(GetNbComposites() > 0);

    for (int i_area = 0; i_area < GetNbComposites(); i_area++) {
        if (i_area == 0) {
            Xaxis = GetXaxisComposite(i_area);
        } else if (i_area == 4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetYmin() == GetComposite(i_area - 1).GetYmin()) {
                Array1DDouble Xaxisbis = GetXaxisComposite(i_area);

                if (Xaxis[0] == 0) {
                    Array1DDouble Xaxisfinal(Xaxisbis.size() + Xaxis.size() - 1);
                    Xaxisfinal.head(Xaxisbis.size()) = Xaxisbis;
                    for (int i = 1; i < Xaxis.size(); i++) {
                        Xaxisfinal[Xaxisbis.size() - 1 + i] = Xaxis[i] + m_axisXmax;
                    }
                    return Xaxisfinal;
                } else {
                    Array1DDouble Xaxisfinal(Xaxisbis.size() + Xaxis.size());
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

Array1DDouble asGeoAreaCompositeGrid::GetYaxis() const
{
    Array1DDouble Yaxis;

    wxASSERT(GetNbComposites() > 0);

    for (int i_area = 0; i_area < GetNbComposites(); i_area++) {
        if (i_area == 0) {
            Yaxis = GetYaxisComposite(i_area);
        } else if (i_area == 4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetXmin() == GetComposite(i_area - 1).GetXmin()) {
                asLogError(_("This function has not been tested"));

                Array1DDouble Yaxisbis = GetYaxisComposite(i_area);

                if (Yaxis[0] == 0) {
                    Array1DDouble Yaxisfinal(Yaxisbis.size() + Yaxis.size() - 1);
                    Yaxisfinal.head(Yaxisbis.size()) = Yaxisbis;
                    for (int i = 1; i < Yaxis.size(); i++) {
                        Yaxisfinal[Yaxisbis.size() - 1 + i] = Yaxis[i] + m_axisYmax;
                    }
                    return Yaxisfinal;
                } else {
                    Array1DDouble Yaxisfinal(Yaxisbis.size() + Yaxis.size());
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

    Array1DDouble xAxis = GetXaxisComposite(0);
    cornerLR1.x = xAxis[xAxis.size()-2];
    cornerUR1.x = xAxis[xAxis.size()-2];

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

    m_lastRowInComposite = true;
}
