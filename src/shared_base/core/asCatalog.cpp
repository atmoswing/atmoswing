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
 
#include "asCatalog.h"

#include "wx/fileconf.h"

#include <asTime.h>
#include <asFileXml.h>


asCatalog::asCatalog(const wxString &alternateFilePath)
{
    m_CatalogFilePath = alternateFilePath;

    // Initialize some data
    m_SetId = wxEmptyString;
    m_Name = wxEmptyString;
    m_Description = wxEmptyString;
    m_TimeZoneHours = 0;
    m_TimeStepHours = 0;
    m_FirstTimeStepHour = 0;
    m_DataPath = wxEmptyString;
    m_Start = 0;
    m_End = 0;

}

asCatalog::~asCatalog()
{
    //dtor
}

double asCatalog::ConvertStringToDatasetDate(const wxString &date_s, int InSerie, float TimeZone, double TimeStepHours, double FirstHour)
{
    if (date_s.IsEmpty()) return NaNDouble;

    // Convert the string into a date
    double date = asTime::GetTimeFromString(date_s, guess);

    // Change units to work in MJD
    double TimeStepDays = TimeStepHours/24;
    double FirstHourDays = FirstHour/24;

    // Add the timezone
    date -= TimeZone/24;

    // Get the day
    int day = floor(date);
    double hour = date-day;

    // Make the date match the dataset definition
    int hourratio = 0;
    if(InSerie == asSERIE_BEGINNING)
    {
        hourratio = ceil((hour-FirstHourDays)/TimeStepDays);
    } else {
        hourratio = floor((hour-FirstHourDays)/TimeStepDays);
    }

    // Build up the final date
    double resdate = day;
    resdate += FirstHourDays + hourratio*TimeStepDays;

    return resdate;
}
