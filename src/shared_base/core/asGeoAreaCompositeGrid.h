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
 
#ifndef asGeoAreaCompositeGrid_H
#define asGeoAreaCompositeGrid_H

#include <asIncludes.h>
#include <asGeoAreaComposite.h>

class asGeoAreaCompositeGrid: public asGeoAreaComposite // Abstract class
{
public:
    enum GridType //!< Define available parameters sets (for the GUI)
    {
        Regular,
        GaussianT62
    };

	asGeoAreaCompositeGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(CoordSys coosys, double Umin, double Uwidth, double Vmin, double Vwidth, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(CoordSys coosys, float Level = asNONE, float Height = asNONE);


    static asGeoAreaCompositeGrid* GetInstance(CoordSys coosys, const wxString &type, double Umin, int Uptsnb, double Ustep, double Vmin, int Vptsnb, double Vstep, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    virtual bool GridsOverlay(asGeoAreaCompositeGrid *otherarea) = 0;


    GridType GetGridType()
    {
        return m_GridType;
    }

    wxString GetGridTypeString()
    {
        switch (m_GridType)
        {
            case (Regular):
                return "Regular";
            case (GaussianT62):
                return "GaussianT62";
            default:
                return "Not found";
        }

        return "Not found";
    }

    virtual double GetUstep() = 0;
    virtual double GetVstep() = 0;

    /** Get the U axis for the given composite
     * \param compositeNb The desired composite
     * \return The axis built on the boundaries and the step
     */
    virtual Array1DDouble GetUaxisComposite(int compositeNb) = 0;

    /** Get the V axis for the given composite
     * \param compositeNb The desired composite
     * \return The axis built on the boundaries and the step
     */
    virtual Array1DDouble GetVaxisComposite(int compositeNb) = 0;

    /** Get the size of the U axis for the given composite
     * \param compositeNb The desired composite
     * \return The size of the axis
     */
    virtual int GetUaxisCompositePtsnb(int compositeNb) = 0;

    /** Get the size of the V axis for the given composite
     * \param compositeNb The desired composite
     * \return The size of the axis
     */
    virtual int GetVaxisCompositePtsnb(int compositeNb) = 0;

    /** Get the width of the U axis for the given composite
     * \param compositeNb The desired composite
     * \return The width of the axis
     */
    virtual double GetUaxisCompositeWidth(int compositeNb) = 0;

    /** Get the width of the V axis for the given composite
     * \param compositeNb The desired composite
     * \return The width of the axis
     */
    virtual double GetVaxisCompositeWidth(int compositeNb) = 0;

    /** Get the start value of the U axis for the given composite
     * \param compositeNb The desired composite
     * \return The start value of the axis
     */
    virtual double GetUaxisCompositeStart(int compositeNb) = 0;

    /** Get the start value of the V axis for the given composite
     * \param compositeNb The desired composite
     * \return The start value of the axis
     */
    virtual double GetVaxisCompositeStart(int compositeNb) = 0;

    /** Get the last value of the U axis for the given composite
     * \param compositeNb The desired composite
     * \return The last value of the axis
     */
    virtual double GetUaxisCompositeEnd(int compositeNb) = 0;

    /** Get the last value of the V axis for the given composite
     * \param compositeNb The desired composite
     * \return The last value of the axis
     */
    virtual double GetVaxisCompositeEnd(int compositeNb) = 0;

    /** Get the total size of the U axis
     * \return The total size of the axis
     */
    int GetUaxisPtsnb();

    /** Get the total size of the V axis
     * \return The total size of the axis
     */
    int GetVaxisPtsnb();

    /** Get the total width of the U axis
     * \return The total width of the axis
     */
    double GetUaxisWidth();

    /** Get the total width of the V axis
     * \return The total width of the axis
     */
    double GetVaxisWidth();

    Array1DDouble GetUaxis();

    Array1DDouble GetVaxis();


protected:
    GridType m_GridType;

private:

};

#endif // asGeoAreaCompositeGrid_H
