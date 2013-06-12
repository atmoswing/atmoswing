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
 
#ifndef ASGEO_H
#define ASGEO_H

#include <proj_api.h>

#include <asIncludes.h>

class asGeo: public wxObject
{
public:

    /** Default constructor */
    asGeo(CoordSys val);

    /** Default destructor */
    virtual ~asGeo();

    /** Access m_CoordSys
     * \return The current value of m_CoordSys
     */
    CoordSys GetCoordSys()
    {
        return m_CoordSys;
    }

    double GetAxisUmin()
    {
        return m_AxisUmin;
    }

    double GetAxisUmax()
    {
        return m_AxisUmax;
    }

    double GetAxisVin()
    {
        return m_AxisVmin;
    }

    double GetAxisVmax()
    {
        return m_AxisVmax;
    }

    /** Access m_CoordSys information
     * \return A description of m_CoordSys
     */
    wxString GetCoordSysInfo();

protected:
    CoordSys m_CoordSys; //!< Member variable "m_CoordSys"
    double m_AxisUmin; //!< Member variable "m_AxisUmin"
    double m_AxisUmax; //!< Member variable "m_AxisUmax"
    double m_AxisVmin; //!< Member variable "m_AxisVmin"
    double m_AxisVmax; //!< Member variable "m_AxisVmax"

    /** Initialization */
    void InitBounds();

    /** Check the point coordinates
     * \param Point The point to check
     */
    bool CheckPoint(Coo &Point, int ChangesAllowed = asEDIT_FORBIDEN);

    /** Projection transform : From the current to the destination
     * \param newcoordsys The destination projection
     * \param coo_src The coordinates in the source projection
     */
    Coo ProjTransform(CoordSys newcoordsys, Coo coo_src);

private:
    /** Projection transform : From WGS84 to CH1903
     * \param coo_src The coordinates in the source projection
     * \return The coordinates in the destination projection
     */
    static Coo ProjWGS84toCH1903(Coo coo_src);

    /** Projection transform : From CH1903 to WGS84
     * \param coo_src The coordinates in the source projection
     * \return The coordinates in the destination projection
     */
    static Coo ProjCH1903toWGS84(Coo coo_src);

    /** Projection transform : From WGS84 to CH1903+
     * \param coo_src The coordinates in the source projection
     * \return The coordinates in the destination projection
     */
    static Coo ProjWGS84toCH1903p(Coo coo_src);

    /** Projection transform : From CH1903+ to WGS84
     * \param coo_src The coordinates in the source projection
     * \return The coordinates in the destination projection
     */
    static Coo ProjCH1903ptoWGS84(Coo coo_src);

    /** Projection transform : From CH1903+ to CH1903
     * \param coo_src The coordinates in the source projection
     * \return The coordinates in the destination projection
     */
    static Coo ProjCH1903ptoCH1903(Coo coo_src);

    /** Projection transform : From CH1903 to CH1903+
     * \param coo_src The coordinates in the source projection
     * \return The coordinates in the destination projection
     */
    static Coo ProjCH1903toCH1903p(Coo coo_src);

};

#endif
