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
 
#include "asGeoAreaCompositeRegularGrid.h"

asGeoAreaCompositeRegularGrid::asGeoAreaCompositeRegularGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, double Xstep, double Ystep, float Level, float Height, int flatAllowed)
:
asGeoAreaCompositeGrid(CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    m_GridType = Regular;
    m_Xstep = Xstep;
    m_Ystep = Ystep;

    if(!IsOnGrid(Xstep, Ystep)) asThrowException(_("The given area does not match a grid."));
}

asGeoAreaCompositeRegularGrid::asGeoAreaCompositeRegularGrid(double Xmin, double Xwidth, double Xstep, double Ymin, double Ywidth, double Ystep, float Level, float Height, int flatAllowed)
:
asGeoAreaCompositeGrid(Xmin, Xwidth, Ymin, Ywidth, Level, Height, flatAllowed)
{
    m_GridType = Regular;
    m_Xstep = Xstep;
    m_Ystep = Ystep;

    if(!IsOnGrid(Xstep, Ystep)) asThrowException(_("The given area does not match a grid."));
}

asGeoAreaCompositeRegularGrid::~asGeoAreaCompositeRegularGrid()
{
    //dtor
}

bool asGeoAreaCompositeRegularGrid::GridsOverlay(asGeoAreaCompositeGrid *otherarea)
{
    if (otherarea->GetGridType()!=Regular) return false;
    asGeoAreaCompositeRegularGrid* otherareaRegular(dynamic_cast<asGeoAreaCompositeRegularGrid*>(otherarea));
    if (GetXstep()!=otherareaRegular->GetXstep()) return false;
    if (GetYstep()!=otherareaRegular->GetYstep()) return false;
    return true;
}

Array1DDouble asGeoAreaCompositeRegularGrid::GetXaxisComposite(int compositeNb)
{
    // Get axis size
    int size = GetXaxisCompositePtsnb(compositeNb);
    Array1DDouble Xaxis = Array1DDouble(size);

    // Build array
    double Xmin = GetComposite(compositeNb).GetXmin();
    if (compositeNb==0) // Left border
    {
        double Xmax = GetComposite(compositeNb).GetXmax();
        double restovers = Xmax-Xmin-m_Xstep*(size-1);
        Xmin += restovers;
    }

    for (int i=0; i<size; i++)
    {
        Xaxis(i) = Xmin+(double)i*m_Xstep;
    }
    //wxASSERT_MSG(Xaxis(size-1)==GetComposite(compositeNb).GetXmax(), wxString::Format("Xaxis(size-1)=%f, GetComposite(%d).GetXmax()=%f", Xaxis(size-1), compositeNb, GetComposite(compositeNb).GetXmax()));  // Not always true

    return Xaxis;
}

Array1DDouble asGeoAreaCompositeRegularGrid::GetYaxisComposite(int compositeNb)
{
    // Get axis size
    int size = GetYaxisCompositePtsnb(compositeNb);
    Array1DDouble Yaxis = Array1DDouble(size);

    // Build array
    double Ymin = GetComposite(compositeNb).GetYmin();
// FIXME (Pascal#3#): Check the compositeNb==0 in this case
    if (compositeNb==0) // Not sure...
    {
        double Ymax = GetComposite(compositeNb).GetYmax();
        double restovers = Ymax-Ymin-m_Ystep*(size-1);
        Ymin += restovers;
    }

    for (int i=0; i<size; i++)
    {
        Yaxis(i) = Ymin+i*m_Ystep;
    }
    //wxASSERT(Yaxis(size-1)==GetComposite(compositeNb).GetYmax()); // Not always true

    return Yaxis;
}

int asGeoAreaCompositeRegularGrid::GetXaxisCompositePtsnb(int compositeNb)
{
    double diff = abs((GetComposite(compositeNb).GetXmax()-GetComposite(compositeNb).GetXmin()))/m_Xstep;
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
}

int asGeoAreaCompositeRegularGrid::GetYaxisCompositePtsnb(int compositeNb)
{
    double diff = abs((GetComposite(compositeNb).GetYmax()-GetComposite(compositeNb).GetYmin()))/m_Ystep;
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
}

double asGeoAreaCompositeRegularGrid::GetXaxisCompositeWidth(int compositeNb)
{
    return abs(GetComposite(compositeNb).GetXmax()-GetComposite(compositeNb).GetXmin());
}

double asGeoAreaCompositeRegularGrid::GetYaxisCompositeWidth(int compositeNb)
{
    return abs(GetComposite(compositeNb).GetYmax()-GetComposite(compositeNb).GetYmin());
}

double asGeoAreaCompositeRegularGrid::GetXaxisCompositeStart(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetXmin();
    }

    // If multiple composites
    if(compositeNb==0) // from 0
    {
        // Composites are not forced on the grid. So we may need to adjust the split of the longitudes axis.
        double dX = abs(GetComposite(1).GetXmax()-GetComposite(1).GetXmin());

        if(fmod(dX, m_Xstep)<0.000001)
        {
            return GetComposite(compositeNb).GetXmin();
        }
        else
        {
            double rest = fmod(dX, m_Xstep);
            return m_Xstep-rest;
        }
    }
    else if (compositeNb==1) // to 360
    {
        return GetComposite(compositeNb).GetXmin();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeRegularGrid::GetYaxisCompositeStart(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetYmin();
    }

    // If multiple composites
    if(compositeNb==0) // from 0
    {
        return GetComposite(compositeNb).GetYmin();
    }
    else if (compositeNb==1) // to 360
    {
        return GetComposite(compositeNb).GetYmin();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeRegularGrid::GetXaxisCompositeEnd(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetXmax();
    }

    // If multiple composites
    if(compositeNb==1) // to 360
    {
        // Composites are not forced on the grid. So we may need to adjust the split of the longitudes axis.
        double dX = abs(GetComposite(1).GetXmax()-GetComposite(1).GetXmin());
        double rest = fmod(dX, m_Xstep);
        if(rest<0.000001)
        {
            return GetComposite(compositeNb).GetXmax();
        }
        else
        {
            return GetComposite(compositeNb).GetXmax()-rest;
        }
    }
    else if (compositeNb==0) // from 0
    {
        return GetComposite(compositeNb).GetXmax();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeRegularGrid::GetYaxisCompositeEnd(int compositeNb)
{
    // If only one composite
    if(GetNbComposites()==1)
    {
        return GetComposite(compositeNb).GetYmax();
    }

    // If multiple composites
    if(compositeNb==1) // to 360
    {
        return GetComposite(compositeNb).GetYmax();
    }
    else if (compositeNb==0) // from 0
    {
        return GetComposite(compositeNb).GetYmax();
    }
    else
    {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

bool asGeoAreaCompositeRegularGrid::IsOnGrid(double step)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(GetXaxisWidth(),step))>0.0000001) return false;
    if (abs(fmod(GetYaxisWidth(),step))>0.0000001) return false;

    return true;
}

bool asGeoAreaCompositeRegularGrid::IsOnGrid(double stepX, double stepY)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(GetXaxisWidth(),stepX))>0.0000001) return false;
    if (abs(fmod(GetYaxisWidth(),stepY))>0.0000001) return false;

    return true;
}
