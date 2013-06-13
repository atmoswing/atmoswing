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
 
#ifndef asGeoAreaRegularGrid_H
#define asGeoAreaRegularGrid_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaRegularGrid: public asGeoArea
{
public:

    /** Default constructor
     * \param coosys The coordinate system
     * \param CornerUL The coordinates of the upper left corner
     * \param CornerUR The coordinates of the upper right corner
     * \param CornerLL The coordinates of the lower left corner
     * \param CornerLR The coordinates of the lower right corner
     * \param Ustep The step according to the U axis
     * \param Vstep The step according to the v axis
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaRegularGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, double Ustep, double Vstep, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    /** Alternative constructor
     * \param coosys The coordinate system
     * \param Umin The left border
     * \param Uwidth The size on U axis
     * \param Ustep The step according to the U axis
     * \param Vmin The left border
     * \param Vwidth The size on V axis
     * \param Vstep The step according to the v axis
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaRegularGrid(CoordSys coosys, double Umin, double Uwidth, double Ustep, double Vmin, double Vwidth, double Vstep, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    /** Default destructor */
    virtual ~asGeoAreaRegularGrid();

    /** Access m_Ustep
     * \return The current value of m_Ustep
     */
    double GetUstep()
    {
        return m_Ustep;
    }

    /** Access m_Vstep
     * \return The current value of m_Vstep
     */
    double GetVstep()
    {
        return m_Vstep;
    }

    /** Get the size of the U axis
     * \return The size of the U axis
     */
    int GetUaxisPtsnb();

    /** Get the size of the V axis
     * \return The size of the V axis
     */
    int GetVaxisPtsnb();

    /** Get the U axis
     * \return The axis built on the boundaries and the step
     */
    Array1DDouble GetUaxis();

    /** Get the V axis
     * \param step The step of the desired axis
     * \return The axis built on the boundaries and the step
     */
    Array1DDouble GetVaxis();

protected:

private:
    double m_Ustep; //!< Member variable "m_Ustep"
    double m_Vstep; //!< Member variable "m_Vstep"

    /** Tells if the area is a straight square compatible with the given step or not
     * \return True if the area is a straight square compatible with the given step
     */
    bool IsOnGrid(double step);

    /** Tells if the area is a straight square compatible with the given steps or not
     * \return True if the area is a straight square compatible with the given steps
     */
    bool IsOnGrid(double stepU, double stepV);
};

#endif // asGeoAreaRegularGrid_H
