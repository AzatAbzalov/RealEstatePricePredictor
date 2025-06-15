document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById('apartmentForm');

    form.addEventListener('submit', function (event) {
        const area = parseFloat(document.querySelector('[name="Area"]').value);
        const living = parseFloat(document.querySelector('[name="Living area"]').value);
        const floor = parseInt(document.querySelector('[name="Floor"]').value);
        const totalFloors = parseInt(document.querySelector('[name="Number of floors"]').value);

        if (living > area) {
            alert("Жилая площадь не может быть больше общей.");
            event.preventDefault();
            return;
        }

        if (floor > totalFloors) {
            alert("Этаж не может быть выше количества этажей в доме.");
            event.preventDefault();
            return;
        }
    });
});
