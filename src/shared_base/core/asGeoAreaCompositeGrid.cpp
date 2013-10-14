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
 
#include "asGeoAreaCompositeGrid.h"
#include "asGeoAreaCompositeRegularGrid.h"
#include "asGeoAreaGaussianGrid.h"
#include "asGeoAreaCompositeGaussianGrid.h"

asGeoAreaCompositeGrid* asGeoAreaCompositeGrid::GetInstance(CoordSys coosys, const wxString &type, double Umin, int Uptsnb, double Ustep, double Vmin, int Vptsnb, double Vstep, float Level, float Height, int flatAllowed )
{
    // If empty, set Regular.
    if (type.IsEmpty())
    {
        asLogMessage(_("The given grid type is empty. A regular grid has been considered."));
        double Uwidth = (double)(Uptsnb-1)*Ustep;
        double Vwidth = (double)(Vptsnb-1)*Vstep;
        asGeoAreaCompositeGrid* area = new asGeoAreaCompositeRegularGrid(coosys, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep, Level, Height, flatAllowed);
        return area;
    }
    else if (type.IsSameAs("Regular", false))
    {
        double Uwidth = (double)(Uptsnb-1)*Ustep;
        double Vwidth = (double)(Vptsnb-1)*Vstep;
        asGeoAreaCompositeGrid* area = new asGeoAreaCompositeRegularGrid(coosys, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep, Level, Height, flatAllowed);
        return area;
    }
    else if (type.IsSameAs("GaussianT62", false))
    {
        asGeoAreaGaussianGrid::GaussianGridType gaussianType = asGeoAreaGaussianGrid::T62;
        asGeoAreaCompositeGrid* area = new asGeoAreaCompositeGaussianGrid(coosys, Umin, Uptsnb, Vmin, Vptsnb, gaussianType, Level, Height, flatAllowed);
        return area;
    }
    else
    {
        asLogError(wxString::Format(_("Given grid type: %s"), type.c_str()));
        asThrowException("The given grid type doesn't correspond to any existing option.");
    }

    return NULL;
}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level, float Height, int flatAllowed)
:
asGeoAreaComposite(coosys, CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{

}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(CoordSys coosys, double Umin, double Uwidth, double Vmin, double Vwidth, float Level, float Height, int flatAllowed)
:
asGeoAreaComposite(coosys, Umin, Uwidth, Vmin, Vwidth, Level, Height, flatAllowed)
{

}

asGeoAreaCompositeGrid::asGeoAreaCompositeGrid(CoordSys coosys, float Level, float Height)
:
asGeoAreaComposite(coosys, Level, Height)
{

}

int asGeoAreaCompositeGrid::GetUaxisPtsnb()
{
    int ptsLon = 0;

    for (int i_area = 0; i_area<GetNbComposites(); i_area++)
    {
        if (i_area==0)
        {
            ptsLon += GetUaxisCompositePtsnb(i_area);
        } else if (i_area==4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetVmin() == GetComposite(i_area-1).GetVmin())
            {
                if (GetUaxisCompositeEnd(i_area) == m_AxisUmax)
                {
                    ptsLon += GetUaxisCompositePtsnb(i_area)-1;
                }
                else
                {
                    ptsLon += GetUaxisCompositePtsnb(i_area);
                }
            }
        }
    }

    return ptsLon;
}

int asGeoAreaCompositeGrid::GetVaxisPtsnb()
{
    int ptsLat = 0;

    for (int i_area = 0; i_area<GetNbComposites(); i_area++)
    {
        if (i_area==0)
        {
            ptsLat += GetVaxisCompositePtsnb(i_area);
        } else if (i_area==4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetUmin() == GetComposite(i_area-1).GetUmin())
            {
                if (GetVaxisCompositeEnd(i_area) == m_AxisVmax)
                {
                    ptsLat += GetVaxisCompositePtsnb(i_area)-1;
                }
                else
                {
                    ptsLat += GetVaxisCompositePtsnb(i_area);
                }
            }
        }
    }

    return ptsLat;
}

double asGeoAreaCompositeGrid::GetUaxisWidth()
{
    double widthLon = 0;

    for (int i_area = 0; i_area<GetNbComposites(); i_area++)
    {
        if (i_area==0)
        {
            widthLon += GetUaxisCompositeWidth(i_area);
        } else if (i_area==4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetVmin() == GetComposite(i_area-1).GetVmin())
            {
                widthLon += GetUaxisCompositeWidth(i_area);
            }
        }
    }

    return widthLon;
}

double asGeoAreaCompositeGrid::GetVaxisWidth()
{
    double widthLat = 0;

    for (int i_area = 0; i_area<GetNbComposites(); i_area++)
    {
        if (i_area==0)
        {
            widthLat += GetVaxisCompositeWidth(i_area);
        } else if (i_area==4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetUmin() == GetComposite(i_area-1).GetUmin())
            {
                widthLat += GetVaxisCompositeWidth(i_area);
            }
        }
    }

    return widthLat;
}

Array1DDouble asGeoAreaCompositeGrid::GetUaxis()
{
    Array1DDouble Uaxis;

    wxASSERT(GetNbComposites()>0);

    for (int i_area = 0; i_area<GetNbComposites(); i_area++)
    {
        if (i_area==0)
        {
            Uaxis = GetUaxisComposite(i_area);
        } else if (i_area==4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetVmin() == GetComposite(i_area-1).GetVmin())
            {
                Array1DDouble Uaxisbis = GetUaxisComposite(i_area);

                if (Uaxis[0]==0)
                {
                    Array1DDouble Uaxisfinal(Uaxisbis.size()+Uaxis.size()-1);
                    Uaxisfinal.head(Uaxisbis.size()) = Uaxisbis;
                    for (int i=1; i<Uaxis.size(); i++)
                    {
                        Uaxisfinal[Uaxisbis.size()-1+i] = Uaxis[i]+m_AxisUmax;
                    }
                    return Uaxisfinal;
                }
                else
                {
                    Array1DDouble Uaxisfinal(Uaxisbis.size()+Uaxis.size());
                    Uaxisfinal.head(Uaxisbis.size()) = Uaxisbis;
                    for (int i=0; i<Uaxis.size(); i++)
                    {
                        Uaxisfinal[Uaxisbis.size()+i] = Uaxis[i]+m_AxisUmax;
                    }
                    return Uaxisfinal;
                }
            }
        }
    }

    return Uaxis;
}

Array1DDouble asGeoAreaCompositeGrid::GetVaxis()
{
    Array1DDouble Vaxis;

    wxASSERT(GetNbComposites()>0);

    for (int i_area = 0; i_area<GetNbComposites(); i_area++)
    {
        if (i_area==0)
        {
            Vaxis = GetVaxisComposite(i_area);
        } else if (i_area==4) {
            // Do nothing here
        } else {
            if (GetComposite(i_area).GetUmin() == GetComposite(i_area-1).GetUmin())
            {
                asLogError(_("This function has not been tested"));

                Array1DDouble Vaxisbis = GetVaxisComposite(i_area);

                if (Vaxis[0]==0)
                {
                    Array1DDouble Vaxisfinal(Vaxisbis.size()+Vaxis.size()-1);
                    Vaxisfinal.head(Vaxisbis.size()) = Vaxisbis;
                    for (int i=1; i<Vaxis.size(); i++)
                    {
                        Vaxisfinal[Vaxisbis.size()-1+i] = Vaxis[i]+m_AxisVmax;
                    }
                    return Vaxisfinal;
                }
                else
                {
                    Array1DDouble Vaxisfinal(Vaxisbis.size()+Vaxis.size());
                    Vaxisfinal.head(Vaxisbis.size()) = Vaxisbis;
                    for (int i=0; i<Vaxis.size(); i++)
                    {
                        Vaxisfinal[Vaxisbis.size()+i] = Vaxis[i]+m_AxisVmax;
                    }
                    return Vaxisfinal;
                }
            }
        }
    }

    return Vaxis;
}
