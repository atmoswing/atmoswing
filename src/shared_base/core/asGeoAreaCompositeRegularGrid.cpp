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
 
#include "asGeoAreaCompositeRegularGrid.h"

asGeoAreaCompositeRegularGrid::asGeoAreaCompositeRegularGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, double Ustep, double Vstep, float Level, float Height, int flatAllowed)
:
asGeoAreaCompositeGrid(coosys, CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    m_GridType = Regular;
    m_Ustep = Ustep;
    m_Vstep = Vstep;

    if(!IsOnGrid(Ustep, Vstep)) asThrowException(_("The given area does not match a grid."));
}

asGeoAreaCompositeRegularGrid::asGeoAreaCompositeRegularGrid(CoordSys coosys, double Umin, double Uwidth, double Ustep, double Vmin, double Vwidth, double Vstep, float Level, float Height, int flatAllowed)
:
asGeoAreaCompositeGrid(coosys, Umin, Uwidth, Vmin, Vwidth, Level, Height, flatAllowed)
{
    m_GridType = Regular;
    m_Ustep = Ustep;
    m_Vstep = Vstep;

    if(!IsOnGrid(Ustep, Vstep)) asThrowException(_("The given area does not match a grid."));
}

asGeoAreaCompositeRegularGrid::~asGeoAreaCompositeRegularGrid()
{
    //dtor
}

bool asGeoAreaCompositeRegularGrid::GridsOverlay(asGeoAreaCompositeGrid *otherarea)
{
    if (otherarea->GetGridType()!=Regular) return false;
    asGeoAreaCompositeRegularGrid *otherareaRegular = (asGeoAreaCompositeRegularGrid*) otherarea;
    if (GetUstep()!=otherareaRegular->GetUstep()) return false;
    if (GetVstep()!=otherareaRegular->GetVstep()) return false;
    return true;
}

Array1DDouble asGeoAreaCompositeRegularGrid::GetUaxisComposite(int compositeNb)
{
    // Get axis size
    int size = GetUaxisCompositePtsnb(compositeNb);
    Array1DDouble Uaxis = Array1DDouble(size);

    // Build array
    double umin = GetComposite(compositeNb).GetUmin();
    if (compositeNb==0) // Left border
    {
        double umax = GetComposite(compositeNb).GetUmax();
        double restovers = umax-umin-m_Ustep*(size-1);
        umin += restovers;
    }

    for (int i=0; i<size; i++)
    {
        Uaxis(i) = umin+(double)i*m_Ustep;
    }
    //wxASSERT_MSG(Uaxis(size-1)==GetComposite(compositeNb).GetUmax(), wxString::Format("Uaxis(size-1)=%f, GetComposite(%d).GetUmax()=%f", Uaxis(size-1), compositeNb, GetComposite(compositeNb).GetUmax()));  // Not always true

    return Uaxis;
}

Array1DDouble asGeoAreaCompositeRegularGrid::GetVaxisComposite(int compositeNb)
{
    // Get axis size
    int size = GetVaxisCompositePtsnb(compositeNb);
    Array1DDouble Vaxis = Array1DDouble(size);

    // Build array
    double vmin = GetComposite(compositeNb).GetVmin();
// FIXME (Pascal#3#): Check the compositeNb==0 in this case
    if (compositeNb==0) // Not sure...
    {
        double vmax = GetComposite(compositeNb).GetVmax();
        double restovers = vmax-vmin-m_Vstep*(size-1);
        vmin += restovers;
    }

    for (int i=0; i<size; i++)
    {
        Vaxis(i) = vmin+i*m_Vstep;
    }
    //wxASSERT(Vaxis(size-1)==GetComposite(compositeNb).GetVmax()); // Not always true

    return Vaxis;
}

int asGeoAreaCompositeRegularGrid::GetUaxisCompositePtsnb(int compositeNb)
{
    double diff = abs((GetComposite(compositeNb).GetUmax()-GetComposite(compositeNb).GetUmin()))/m_Ustep;
    double size;
    double rest = modf (diff , &size);

    if(compositeNb==0) // from 0
    {
        size += 1;
    }
    else if (compositeNb==1) // to 360
    {
        size += 1;
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }

    if (rest<0.0000001 || rest>0.999999) //Precision issue
    {
        return size + asTools::Round(rest);
    }
    else
    {
        return size;
    }

    return asNOT_VALID;
}

int asGeoAreaCompositeRegularGrid::GetVaxisCompositePtsnb(int compositeNb)
{
    double diff = abs((GetComposite(compositeNb).GetVmax()-GetComposite(compositeNb).GetVmin()))/m_Vstep;
    double size;
    double rest = modf (diff , &size);
    size += 1;

    if (rest<0.0000001 || rest>0.999999) //Precision issue
    {
        return size + asTools::Round(rest);
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }

    return asNOT_VALID;
}

double asGeoAreaCompositeRegularGrid::GetUaxisCompositeWidth(int compositeNb)
{
    return abs(GetComposite(compositeNb).GetUmax()-GetComposite(compositeNb).GetUmin());
}

double asGeoAreaCompositeRegularGrid::GetVaxisCompositeWidth(int compositeNb)
{
    return abs(GetComposite(compositeNb).GetVmax()-GetComposite(compositeNb).GetVmin());
}

double asGeoAreaCompositeRegularGrid::GetUaxisCompositeStart(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetUmin();
    }

    // If multiple composites
    if(compositeNb==0) // from 0
    {
        // Composites are not forced on the grid. So we may need to adjust the split of the longitudes axis.
        double dU = abs(GetComposite(1).GetUmax()-GetComposite(1).GetUmin());

        if(fmod(dU, m_Ustep)<0.000001)
        {
            return GetComposite(compositeNb).GetUmin();
        }
        else
        {
            double rest = fmod(dU, m_Ustep);
            return m_Ustep-rest;
        }
    }
    else if (compositeNb==1) // to 360
    {
        return GetComposite(compositeNb).GetUmin();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeRegularGrid::GetVaxisCompositeStart(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetVmin();
    }

    // If multiple composites
    if(compositeNb==0) // from 0
    {
        return GetComposite(compositeNb).GetVmin();
    }
    else if (compositeNb==1) // to 360
    {
        return GetComposite(compositeNb).GetVmin();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeRegularGrid::GetUaxisCompositeEnd(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetUmax();
    }

    // If multiple composites
    if(compositeNb==1) // to 360
    {
        // Composites are not forced on the grid. So we may need to adjust the split of the longitudes axis.
        double dU = abs(GetComposite(1).GetUmax()-GetComposite(1).GetUmin());
        double rest = fmod(dU, m_Ustep);
        if(rest<0.000001)
        {
            return GetComposite(compositeNb).GetUmax();
        }
        else
        {
            return GetComposite(compositeNb).GetUmax()-rest;
        }
        return GetComposite(compositeNb).GetUmax();
    }
    else if (compositeNb==0) // from 0
    {
        return GetComposite(compositeNb).GetUmax();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeRegularGrid::GetVaxisCompositeEnd(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetVmax();
    }

    // If multiple composites
    if(compositeNb==1) // to 360
    {
        return GetComposite(compositeNb).GetVmax();
    }
    else if (compositeNb==0) // from 0
    {
        return GetComposite(compositeNb).GetVmax();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

bool asGeoAreaCompositeRegularGrid::IsOnGrid(double step)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(GetUaxisWidth(),step))>0.0000001) return false;
    if (abs(fmod(GetVaxisWidth(),step))>0.0000001) return false;

    return true;
}

bool asGeoAreaCompositeRegularGrid::IsOnGrid(double stepU, double stepV)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(GetUaxisWidth(),stepU))>0.0000001) return false;
    if (abs(fmod(GetVaxisWidth(),stepV))>0.0000001) return false;

    return true;
}
