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
 
#ifndef asGeoAreaGaussianGrid_H
#define asGeoAreaGaussianGrid_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaGaussianGrid: public asGeoArea
{
public:
	enum GaussianGridType
	{
		T62
	};

    /** Default constructor
     * \param coosys The coordinate system
     * \param CornerUL The coordinates of the upper left corner
     * \param CornerUR The coordinates of the upper right corner
     * \param CornerLL The coordinates of the lower left corner
     * \param CornerLR The coordinates of the lower right corner
     * \param type The Gaussian grid type
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaGaussianGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, GaussianGridType type = T62, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    /** Alternative constructor
     * \param coosys The coordinate system
     * \param Umin The left border
     * \param Uptsnb The size on U axis
     * \param Vmin The left border
     * \param Vptsnb The size on V axis
     * \param type The Gaussian grid type
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaGaussianGrid(CoordSys coosys, double Umin, int Uptsnb, double Vmin, int Vptsnb, GaussianGridType type = T62, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    /** Default destructor */
    virtual ~asGeoAreaGaussianGrid();

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
    Array1DDouble m_FullAxisU;
    Array1DDouble m_FullAxisV;

    bool IsOnGrid(const Coo &point);
    bool IsOnGrid(double Ucoord, double Vcoord);

};

#endif // asGeoAreaGaussianGrid_H
